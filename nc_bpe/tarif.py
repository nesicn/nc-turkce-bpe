"""Tokenizer tarifi: özel token'lar, normalizer, ön bölme deseni ve eğitilmemiş tokenizer iskeleti.

Üç sözlük boyutu (32k, 64k, 128k) aynı tarifle kurulur; boyutlar arasında yalnız sözlük boyutu ve 1. aşamanın
sözlüğü (V1) değişir. Gerekçeler: docs/yontem.md
"""
from __future__ import annotations

import json

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# --------------------------------------------------------------------------------------------
# Yayımlanan tokenizer'ların ayarları. Sözlük boyutu özel token'lar DAHİL toplamdır (len(tokenizer)).
# --------------------------------------------------------------------------------------------
HAZIR_AYARLAR = {
    "32k": {"sozluk": 32_000, "v1": 24_000},
    "64k": {"sozluk": 64_000, "v1": 36_000},
    "128k": {"sozluk": 128_000, "v1": 80_000},
}
VARSAYILAN_RAKAM = "uclu"
VARSAYILAN_EKLER = ("sb", "ns")
VARSAYILAN_MIN_FREKANS = 2
MODEL_MAX_LENGTH = 131_072

# --------------------------------------------------------------------------------------------
# Özel token'lar: ID 0..63, bütün boyutlarda aynı. <|...|> biçimi doğal metinde neredeyse hiç geçmez.
# Rezerve token'lar ileride yeniden adlandırılabilir; ID'ler değişmez, tokenizer yeniden eğitilmez.
# --------------------------------------------------------------------------------------------
OZEL_TOKEN_SAYISI = 64
TEMEL_OZEL_TOKENLAR = [
    "<|endoftext|>",    # 0  belge sonu / EOS
    "<|startoftext|>",  # 1  BOS
    "<|pad|>",          # 2  dolgu
    "<|unk|>",          # 3  bilinmeyen (byte düzeyi BPE'de hiç üretilmez)
    "<|mask|>",         # 4  maskeli dil modeli
    "<|im_start|>",     # 5  sohbet mesajı başı (ChatML)
    "<|im_end|>",       # 6  sohbet mesajı sonu (ChatML)
]
OZEL_TOKENLAR = TEMEL_OZEL_TOKENLAR + [
    f"<|reserved_{i}|>" for i in range(OZEL_TOKEN_SAYISI - len(TEMEL_OZEL_TOKENLAR))
]
assert len(OZEL_TOKENLAR) == OZEL_TOKEN_SAYISI == len(set(OZEL_TOKENLAR))

TOKEN_ROLLERI = {
    "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "mask_token": "<|mask|>",
}
UNK_ID = OZEL_TOKENLAR.index("<|unk|>")

# --------------------------------------------------------------------------------------------
# Ön bölme deseni (Hugging Face Split, Oniguruma). Sırayla:
#   a) Türkçe kesme işaretli ek: harf/rakamdan hemen sonra gelen ' veya ’ + harfler → "'da", "'nin", "'te".
#      İngilizce 's|'t|'d kuralı yoktur: "Ankara'da" → "Ankara" + "'da".
#   b) Kelime: isteğe bağlı tek boşluk + harf ve birleşen işaret dizisi. Noktalama kelimeye yapışmaz.
#   c) Rakam (aşağıdaki seçenekler), d) noktalama/sembol dizisi (isteğe bağlı tek boşlukla),
#   e) satır sonları, f) sondaki boşluk dışındaki boşluk dizisi, g) kalan boşluk.
# --------------------------------------------------------------------------------------------
RAKAM_DESENLERI = {
    "tek": r"\p{N}",         # 2026 → 2·0·2·6
    "uclu": r"\p{N}{1,3}",   # 2026 → 202·6
    "hepsi": r"\p{N}+",      # 2026 → 2026
}
_KESME = "['’]"

# Desen ekleri:
#   sb: sayının önündeki tek boşluk sayıya yapışır (" 2026" → " 202"·"6").
#   ns: noktalamanın hemen ardındaki satır sonları noktalamaya yapışır (".\n\n" tek parça).
#   is: birleşen işaretler (ör. emoji varyasyon seçicisi) sembol dizisinde kalır.
ON_BOLME_EKLERI = ("sb", "ns", "is")


def ekleri_duzenle(ekler) -> tuple[str, ...]:
    """Ek listesini doğrular ve sabit sıraya koyar ("ns,sb" → ("sb", "ns"))."""
    if isinstance(ekler, str):
        ekler = [e for e in ekler.split(",") if e]
    ekler = tuple(ekler or ())
    bilinmeyen = [e for e in ekler if e not in ON_BOLME_EKLERI]
    if bilinmeyen or len(set(ekler)) != len(ekler):
        raise ValueError(f"ekler {ON_BOLME_EKLERI} içinden ve tekrarsız olmalı, verilen: {ekler!r}")
    return tuple(e for e in ON_BOLME_EKLERI if e in ekler)


def on_bolme_deseni(rakam: str = VARSAYILAN_RAKAM, ekler=VARSAYILAN_EKLER) -> str:
    if rakam not in RAKAM_DESENLERI:
        raise ValueError(f"rakam {list(RAKAM_DESENLERI)} olmalı, verilen: {rakam!r}")
    ekler = ekleri_duzenle(ekler)
    sayi = ("[ ]?" if "sb" in ekler else "") + RAKAM_DESENLERI[rakam]
    noktalama_sinifi = r"[^\s\p{L}\p{N}]" if "is" in ekler else r"[^\s\p{L}\p{M}\p{N}]"
    noktalama = rf"[ ]?{noktalama_sinifi}+" + (r"[\r\n]*" if "ns" in ekler else "")
    return (
        rf"(?<=[\p{{L}}\p{{M}}\p{{N}}]){_KESME}[\p{{L}}\p{{M}}]+"
        rf"|[ ]?[\p{{L}}\p{{M}}]+"
        rf"|{sayi}"
        rf"|{noktalama}"
        rf"|\s*[\r\n]+"
        rf"|\s+(?!\S)"
        rf"|\s+"
    )


def tokenizer_iskeleti(rakam: str = VARSAYILAN_RAKAM, ekler=VARSAYILAN_EKLER) -> Tokenizer:
    """Eğitilmemiş byte düzeyi BPE tokenizer.

    - Normalizer yalnız NFC: küçük harfe çevirme yok (I/ı, İ/i bozulur), NFKC yok (kayıplı: ﬁ→fi, x²→x2).
    - Byte düzeyi: her byte'ın sözlükte karşılığı var, <unk> hiç oluşmaz.
    - Post-processor yok: tokenizer kendiliğinden BOS/EOS eklemez.
    """
    bolme = pre_tokenizers.Split(Regex(on_bolme_deseni(rakam, ekler)), behavior="isolated", invert=False)
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False, fuse_unk=False, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        bolme, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)])
    tok.decoder = decoders.ByteLevel()
    tok.normalizer = normalizers.NFC()
    tok.post_processor = None
    return tok


def egitici(sozluk_boyutu: int, min_frequency: int = VARSAYILAN_MIN_FREKANS, show_progress: bool = False):
    """Sözlük = 64 özel token + 256 byte (alfabe) + öğrenilen birleştirmeler. Alfabe 256 byte'ın tamamıdır."""
    return trainers.BpeTrainer(
        vocab_size=sozluk_boyutu, min_frequency=min_frequency, special_tokens=list(OZEL_TOKENLAR),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=show_progress,
    )


def iskelet_json(rakam: str = VARSAYILAN_RAKAM, ekler=VARSAYILAN_EKLER) -> dict:
    """Birleştirmesi olmayan tokenizer.json: özel token'lar + 256 byte, tarifin normalizer/ön bölme/decoder'ı.

    Eğitimde yalnız iskelet olarak kullanılır; model (sözlük ve birleştirmeler) `bpe.sozluk_kur` ile yazılır.
    """
    tok = tokenizer_iskeleti(rakam, ekler)
    tok.train_from_iterator([], trainer=egitici(OZEL_TOKEN_SAYISI + 256))
    j = json.loads(tok.to_str())
    if len(j["model"]["vocab"]) != OZEL_TOKEN_SAYISI + 256 or j["model"]["merges"]:
        raise RuntimeError("iskelet beklenen biçimde değil")
    return j


def tarif_ayarlari(j: dict) -> tuple[str, tuple[str, ...]] | None:
    """tokenizer.json'daki ön bölme deseninden (rakam, ekler) ayarını bulur; tarifle üretilmemişse None."""
    import itertools
    pre = j.get("pre_tokenizer") or {}
    parcalar = pre.get("pretokenizers", [pre])
    desen = next((p["pattern"]["Regex"] for p in parcalar if p.get("type") == "Split"), None)
    for rakam in RAKAM_DESENLERI:
        for n in range(len(ON_BOLME_EKLERI) + 1):
            for ekler in itertools.combinations(ON_BOLME_EKLERI, n):
                if on_bolme_deseni(rakam, ekler) == desen:
                    return rakam, ekler
    return None
