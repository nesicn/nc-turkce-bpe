"""Hugging Face paketi ve eğitim sonrası yapısal kontroller."""
from __future__ import annotations

import json
import os
import unicodedata

from tokenizers import Tokenizer

from .tarif import MODEL_MAX_LENGTH, OZEL_TOKEN_SAYISI, OZEL_TOKENLAR, TOKEN_ROLLERI, ekleri_duzenle


def gpt2_bayt_tablosu() -> dict[str, int]:
    """GPT-2'nin byte → görünür karakter eşlemesinin tersi: {karakter: byte} (Ġ → 32, Ċ → 10)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BAYT_TABLOSU = gpt2_bayt_tablosu()


def bayttan_metne(parca: str) -> str:
    """Byte düzeyi gösterimi (Ġ, Ã¼) gerçek metne çevirir."""
    return bytes(_BAYT_TABLOSU[c] for c in parca).decode("utf-8", "replace")


# --------------------------------------------------------------------------------------------
# Hugging Face paketi: tokenizer.json + tokenizer_config.json
# --------------------------------------------------------------------------------------------
def hf_tokenizer(tok: Tokenizer, model_max_length: int = MODEL_MAX_LENGTH):
    """transformers.PreTrainedTokenizerFast sarmalayıcısı; token rolleri ve decode ayarları sabit."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,   # True olursa decode " ." → "." yapar ve kayıpsızlık bozulur
        **TOKEN_ROLLERI,
    )


def hf_paketi_kaydet(tok: Tokenizer, klasor: str, model_max_length: int = MODEL_MAX_LENGTH) -> list[str]:
    """tokenizer.json + tokenizer_config.json yazar.

    transformers 5.x varsayılan olarak tokenizer_class="TokenizersBackend" yazar; bu sınıf transformers 4.x'te yoktur.
    "PreTrainedTokenizerFast" iki sürümde de açılır ve aynı ID'leri verir.
    """
    os.makedirs(klasor, exist_ok=True)
    dosyalar = hf_tokenizer(tok, model_max_length).save_pretrained(klasor)
    yol = os.path.join(klasor, "tokenizer_config.json")
    with open(yol, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return sorted(os.path.basename(d) for d in dosyalar)


# --------------------------------------------------------------------------------------------
# Yapısal kontroller (her eğitimden sonra)
# --------------------------------------------------------------------------------------------
TURK_HARFLERI = list("çğıöşüÇĞİÖŞÜ")
ON_BOLME_ORNEKLERI = {
    # metin: beklenen ön bölme parçaları (rakam="tek", ek yok)
    "Ankara'da": ["Ankara", "'da"],
    "1923'te": ["1", "9", "2", "3", "'te"],
    "Türkiye’nin": ["Türkiye", "’nin"],
    "TBMM'de": ["TBMM", "'de"],
    "Ankara'dır.": ["Ankara", "'dır", "."],
    " 'Merhaba' dedi.": [" '", "Merhaba", "'", " dedi", "."],
    "(Ankara) güzel,": ["(", "Ankara", ")", " güzel", ","],
    "i̇stanbul": ["i̇stanbul"],
    "satır\n\nyeni": ["satır", "\n\n", "yeni"],
}
# (metin, ek VARKEN beklenen, ek YOKKEN beklenen); rakam seçeneğinden bağımsız.
EK_ORNEKLERI = {
    "sb": [("yıl 7 gün", ["yıl", " 7", " gün"], ["yıl", " ", "7", " gün"])],
    "ns": [("bitti.\n\nYeni", ["bitti", ".\n\n", "Yeni"], ["bitti", ".", "\n\n", "Yeni"]),
           ("dur,\r\ngel", ["dur", ",\r\n", "gel"], ["dur", ",", "\r\n", "gel"])],
    "is": [("çok ❤️❤️!", ["çok", " ❤️❤️!"], ["çok", " ❤", "️", "❤", "️", "!"])],
}
GIDIS_DONUS_ORNEKLERI = [
    "Ankara'da 1923'te kuruldu.", "İSTANBUL ışık ILIK İĞNE iğne", "x² ﬁ … ①", "satır\n\n\tsekme\r\n  boşluk  ",
    "😂👨‍👩‍👧 <s> </s> [CLS]", "def f(x):\n    return x ** 2\n", "a b c​d",
    "__label__NOT @USER selam", "https://www.ornek.com.tr/haber?id=5&q=ş", "\x00\x07﻿bom",
]


def _on_bol(tok: Tokenizer, metin: str) -> list[str]:
    n = tok.normalizer.normalize_str(metin)
    return [bayttan_metne(p) for p, _ in tok.pre_tokenizer.pre_tokenize_str(n)]


def yapisal_kontroller(tok: Tokenizer, beklenen_boyut: int, rakam: str, ekler) -> dict:
    """Sözlük boyutu, özel token ID'leri, Türkçe harfler, ön bölme ve gidiş-dönüş. Sorun varsa 'hatalar' doludur."""
    tok = getattr(tok, "backend_tokenizer", tok)
    ekler = ekleri_duzenle(ekler)
    hatalar, sonuc = [], {}
    V = tok.get_vocab_size(with_added_tokens=True)
    sonuc["sozluk_boyutu"] = V
    if V != beklenen_boyut:
        hatalar.append(f"sözlük boyutu {V} ≠ {beklenen_boyut} (veri yetersiz olabilir)")
    vocab = tok.get_vocab(with_added_tokens=True)
    if sorted(vocab.values()) != list(range(V)):
        hatalar.append("ID'ler 0..V-1 bitişik değil")
    yanlis_id = {t: vocab.get(t) for i, t in enumerate(OZEL_TOKENLAR) if vocab.get(t) != i}
    if yanlis_id:
        hatalar.append(f"özel token ID'leri beklenenden farklı: {dict(list(yanlis_id.items())[:5])}")
    eklenen = tok.get_added_tokens_decoder()
    ozel_isaretsiz = [t.content for t in eklenen.values() if not t.special]
    if len(eklenen) != OZEL_TOKEN_SAYISI or ozel_isaretsiz:
        hatalar.append(f"eklenmiş token sayısı {len(eklenen)} (beklenen {OZEL_TOKEN_SAYISI}); özel işaretsiz: {ozel_isaretsiz[:5]}")
    harf = {h: len(tok.encode(h, add_special_tokens=False).ids) for h in TURK_HARFLERI}
    tek_degil = [h for h, n in harf.items() if n != 1]
    if tek_degil:
        hatalar.append(f"tek token olmayan Türkçe harf: {tek_degil}")
    for metin, beklenen in ON_BOLME_ORNEKLERI.items():
        if rakam != "tek" and any(ch.isdigit() for ch in metin):
            continue
        parcalar = _on_bol(tok, metin)
        if parcalar != beklenen:
            hatalar.append(f"ön bölme {metin!r}: {parcalar} ≠ {beklenen}")
    for ek, ornekler in EK_ORNEKLERI.items():
        for metin, ile, ilesiz in ornekler:
            beklenen = ile if ek in ekler else ilesiz
            parcalar = _on_bol(tok, metin)
            if parcalar != beklenen:
                hatalar.append(f"ön bölme ({ek} {'var' if ek in ekler else 'yok'}) {metin!r}: {parcalar} ≠ {beklenen}")
    for m in GIDIS_DONUS_ORNEKLERI:
        geri = tok.decode(tok.encode(m, add_special_tokens=False).ids, skip_special_tokens=False)
        if geri != unicodedata.normalize("NFC", m):
            hatalar.append(f"gidiş-dönüş {m!r} → {geri!r}")
    sonuc["hatalar"] = hatalar
    return sonuc
