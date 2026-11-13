"""Sayımdan BPE eğitimi: düz BPE ve morfem öncelikli iki aşamalı BPE.

İki aşamalı yöntem birleştirmeleri iki aşamada öğrenir; sonuç yine tek bir sıralı birleştirme listesidir (standart byte
düzeyi BPE, `tokenizer.json`):

  1. aşama: En sık kelimeler kesin morfem sınırlarından parçalanır (`morfoloji.birinci_asama_girdisi`). BPE bu parçalar
     üzerinde V1 boyutuna kadar eğitilir; parça sınırını aşan birleştirme öğrenemez.
  2. aşama: Derlemin bütün ön bölme parçaları (bölünmemiş) 1. aşamanın birleştirmeleriyle kodlanır; 1. aşamanın her
     token'ı tek bir özel kullanım alanı (PUA) karakterine eşlenir ve BPE bu karakter dizileri üzerinde, kalan sözlüğü
     dolduracak kadar eğitilmeye devam eder. Öğrenilen birleştirmeler yeniden byte düzeyi dizgelere çevrilir.

Son sözlük = 64 özel token + 256 byte + 1. aşamanın birleştirmeleri + 2. aşamanın yeni dizgeleri.
"""
from __future__ import annotations

import collections
import json
import time

from .paket import gpt2_bayt_tablosu
from .tarif import OZEL_TOKENLAR, VARSAYILAN_MIN_FREKANS

_TABLO = gpt2_bayt_tablosu()
_ALFABE = sorted(_TABLO)                    # 256 byte karakteri, HF eğiticisinin sırasıyla (kod noktası sırası)
_OZEL = len(OZEL_TOKENLAR)

# 1. aşamanın token'ları 2. aşamada birer PUA karakteriyle gösterilir: önce Ek Özel Kullanım Alanı-A
# (U+F0000–U+FFFFD, 65.534 kod noktası), sonra Ek Özel Kullanım Alanı-B (U+100000–U+10FFFD). Karakter olmayan kod
# noktaları (U+FFFFE/F) atlanır. Eşleme kimlik sırasını korur; HF eğiticisi alfabeyi kod noktasına göre sıraladığı
# için eşit sayılı çiftler tek seferlik eğitimdeki gibi seçilir.
PUA_TABAN = 0xF0000
PUA_A_BOYUT = 65_534
PUA_B_TABAN = 0x100000
PUA_B_BOYUT = 65_534


def birlestirme_sayisi(v: int) -> int:
    """Toplam sözlük v'de öğrenilen birleştirme sayısı."""
    return v - _OZEL - 256


def pua_kodu(i: int) -> int:
    """1. aşamanın i. kimliğinin PUA kod noktası (A, sonra B). Sığmazsa ValueError."""
    if i < 0:
        raise ValueError(f"kimlik negatif olamaz: {i}")
    if i < PUA_A_BOYUT:
        return PUA_TABAN + i
    j = i - PUA_A_BOYUT
    if j < PUA_B_BOYUT:
        return PUA_B_TABAN + j
    raise ValueError(f"PUA-A + PUA-B ({PUA_A_BOYUT + PUA_B_BOYUT:,} kod noktası) yetmiyor: kimlik {i:,}")


def pua_karakterleri(n: int) -> list[str]:
    """0..n-1 kimliklerinin PUA karakterleri (sırası kimlik sırasıyla aynı)."""
    if n > PUA_A_BOYUT + PUA_B_BOYUT:
        raise ValueError(f"1. aşamanın sözlüğü ({n:,}) PUA-A + PUA-B'ye ({PUA_A_BOYUT + PUA_B_BOYUT:,}) sığmıyor")
    return [chr(pua_kodu(i)) for i in range(n)]


# --------------------------------------------------------------------------------------------
# Sayımdan BPE (HF tokenizers; sayılar tekrarla verilir)
# --------------------------------------------------------------------------------------------
def _tekrarla(sayim: collections.Counter, parti: int = 200_000):
    """HF eğiticisine (parça, sayı) çiftlerini metin akışı olarak verir: her parça sayısı kadar, boşlukla ayrılmış."""
    buf, n = [], 0
    for p, c in sayim.items():
        while c > 0:
            k = min(c, parti)
            buf.append((p + " ") * k)
            n += k
            c -= k
            if n >= parti:
                yield "".join(buf)
                buf, n = [], 0
    if buf:
        yield "".join(buf)


def sayimdan_bpe(sayim: collections.Counter, sozluk: int, alfabe: list[str] | None,
                 min_frequency: int = VARSAYILAN_MIN_FREKANS):
    """Sayımdan BPE (yalnız model). Girdi parçalarında boşluk yok (byte düzeyi ya da PUA); WhitespaceSplit ile ayrılır.
    Döner: (vocab, merges)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=sozluk, min_frequency=min_frequency, special_tokens=[], show_progress=False,
                             **({"initial_alphabet": alfabe} if alfabe else {}))
    tok.train_from_iterator(_tekrarla(sayim), trainer=tr)
    j = json.loads(tok.to_str())
    return j["model"]["vocab"], [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in j["model"]["merges"]]


# --------------------------------------------------------------------------------------------
# Sözlüğün kurulması
# --------------------------------------------------------------------------------------------
def sozluk_kur(iskelet: dict, merges1, merges2, hedef: int) -> tuple[dict, dict]:
    """İskeletin tokenizer.json'ında yalnız modeli yazar. Sözlük tam `hedef` olana dek önce 1. aşamanın, sonra 2. aşamanın
    birleştirmeleri sırayla eklenir; yinelenen birleştirme çifti atılır. Döner: (tokenizer.json sözlüğü, bilgi)."""
    j = json.loads(json.dumps(iskelet))
    eski = j["model"]["vocab"]
    ozel = sorted((i, t) for t, i in eski.items() if i < _OZEL)
    if [t for _, t in ozel] != list(OZEL_TOKENLAR):
        raise RuntimeError("iskeletin özel token'ları beklenen sırada değil")
    vocab = {t: i for i, t in ozel}
    for c in sorted((c for c in eski if len(c) == 1 and c in _TABLO), key=lambda c: eski[c]):
        vocab[c] = len(vocab)
    if len(vocab) != _OZEL + 256:
        raise RuntimeError("iskeletin byte alfabesi eksik")
    merges, gorulen = [], set()
    cakisma = yinelenen = 0
    for asama, liste in ((1, merges1), (2, merges2)):
        for a, b in liste:
            if len(vocab) >= hedef and (a + b) not in vocab:
                break
            if a not in vocab or b not in vocab:
                raise RuntimeError(f"birleştirmenin parçası sözlükte yok: {a!r} + {b!r}")
            if (a, b) in gorulen:
                yinelenen += 1
                continue
            gorulen.add((a, b))
            merges.append([a, b])
            if a + b in vocab:
                cakisma += asama == 2
            else:
                vocab[a + b] = len(vocab)
    if len(vocab) != hedef:
        raise RuntimeError(f"sözlük {len(vocab)} oldu, {hedef} bekleniyordu (birleştirme yetmedi; veri yetersiz olabilir)")
    j["model"]["vocab"] = vocab
    j["model"]["merges"] = merges
    return j, {"birlestirme": len(merges), "birinci": len(merges1), "ikinci_kullanilan": len(merges) - len(merges1),
               "ikinci_cakisma": cakisma, "yinelenen_cift": yinelenen}


# --------------------------------------------------------------------------------------------
# Düz BPE
# --------------------------------------------------------------------------------------------
def duz_bpe(sayim: collections.Counter, hedef: int, iskelet: dict,
            min_frequency: int = VARSAYILAN_MIN_FREKANS) -> tuple[dict, dict]:
    """Ön bölme sayımından düz byte düzeyi BPE (metinden eğitimle aynı birleştirmeler)."""
    t0 = time.time()
    _, merges = sayimdan_bpe(sayim, hedef - _OZEL, list(_ALFABE), min_frequency)
    if len(merges) != birlestirme_sayisi(hedef):
        raise RuntimeError(f"düz BPE {len(merges)} birleştirme öğrendi, {birlestirme_sayisi(hedef)} bekleniyordu (veri yetersiz)")
    j, bilgi = sozluk_kur(iskelet, merges, [], hedef)
    bilgi.update(v1=None, sure_dk=round((time.time() - t0) / 60, 1))
    return j, bilgi


# --------------------------------------------------------------------------------------------
# İki aşamalı BPE
# --------------------------------------------------------------------------------------------
def birinci_asamalar(girdi: collections.Counter, v1ler, min_frequency: int = VARSAYILAN_MIN_FREKANS) -> tuple[dict, int]:
    """1. aşama en büyük V1 ile bir kez eğitilir; her V1'in birleştirmeleri onun önekidir (BPE açgözlü ve belirlenimci).
    Veri bir V1'e yetmezse o V1 None olur. Döner: ({V1: birleştirmeler | None}, öğrenilen birleştirme sayısı)."""
    en = max(v1ler)
    _, m = sayimdan_bpe(girdi, en - _OZEL, list(_ALFABE), min_frequency)
    out = {}
    for v in sorted(v1ler):
        n = birlestirme_sayisi(v)
        out[v] = m[:n] if len(m) >= n else None
    return out, len(m)


def asama_tokenizeri(merges: list[tuple[str, str]]):
    """Yalnız model: 256 byte + birleştirmeler; ön bölme yok (girdi zaten tek bir byte düzeyi ön bölme parçası).
    Kimlikler HF eğiticisinin vereceği sırayla (alfabe kod noktası sırasıyla, sonra birleştirmeler oluşma sırasıyla)."""
    from tokenizers import Tokenizer, models
    vocab = {c: i for i, c in enumerate(_ALFABE)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return Tokenizer(models.BPE(vocab=vocab, merges=list(merges))), vocab


def ikinci_asama_sayimi(sayim: collections.Counter, merges1: list[tuple[str, str]], parti: int = 50_000):
    """Bütün parçaları 1. aşamayla kodlar ve PUA dizgelerine çevirir. Döner: (PUA sayımı, PUA karakteri → dizge)."""
    tok, vocab = asama_tokenizeri(merges1)
    ters = {i: s for s, i in vocab.items()}
    kar = pua_karakterleri(len(vocab))
    pua = collections.Counter()
    parcalar = list(sayim)
    for i in range(0, len(parcalar), parti):
        p = parcalar[i:i + parti]
        for x, e in zip(p, tok.encode_batch(p, add_special_tokens=False)):
            pua["".join([kar[j] for j in e.ids])] += sayim[x]
    return pua, {kar[i]: s for i, s in ters.items()}


def ikinci_asama(pua: collections.Counter, eslem: dict[str, str], yeni_hedef: int, pay: float = 0.03,
                 min_frequency: int = VARSAYILAN_MIN_FREKANS) -> list[tuple[str, str]]:
    """PUA dizgeleri üzerinde BPE; birleştirmeler byte düzeyi dizgelere çevrilir. Çakışmalar ve yinelenen çiftler
    `sozluk_kur`'da atıldığı için hedeften biraz fazla (pay) öğrenilir."""
    alfabe = sorted({ch for s in pua for ch in s})
    _, merges = sayimdan_bpe(pua, len(alfabe) + int(yeni_hedef * (1 + pay)) + 10, alfabe, min_frequency)
    return [("".join(eslem[c] for c in a), "".join(eslem[c] for c in b)) for a, b in merges]


def iki_asamali(sayim: collections.Counter, merges1, v1: int, iskelet: dict, hedef: int,
                min_frequency: int = VARSAYILAN_MIN_FREKANS, pay: float = 0.03, ilerleme=None) -> tuple[dict, dict]:
    """1. aşamanın birleştirmeleri hazırken 2. aşama ve sözlük. Döner: (tokenizer.json sözlüğü, bilgi)."""
    t0 = time.time()
    if len(merges1) != birlestirme_sayisi(v1):
        raise ValueError("1. aşamanın birleştirme sayısı V1 ile uyuşmuyor")
    pua, eslem = ikinci_asama_sayimi(sayim, merges1)
    farkli2 = len(pua)
    merges2 = ikinci_asama(pua, eslem, hedef - v1, pay=pay, min_frequency=min_frequency)
    del pua
    if ilerleme:
        ilerleme(f"  2. aşama: {len(merges2):,} birleştirme · {(time.time() - t0) / 60:.1f} dk")
    j, bilgi = sozluk_kur(iskelet, merges1, merges2, hedef)
    bilgi.update(v1=v1, ikinci_farkli_parca=farkli2, sure_dk=round((time.time() - t0) / 60, 1))
    return j, bilgi


# --------------------------------------------------------------------------------------------
# İç içelik
# --------------------------------------------------------------------------------------------
def ic_icelik(kucuk: dict, buyuk: dict) -> dict:
    """Küçük sözlüğün büyüğün öneki olup olmadığı: ortak birleştirme öneki ve aynı ID'li token sayısı."""
    mk, mb = kucuk["model"]["merges"], buyuk["model"]["merges"]
    n = 0
    for a, b in zip(mk, mb):
        if list(a) != list(b):
            break
        n += 1
    vk, vb = kucuk["model"]["vocab"], buyuk["model"]["vocab"]
    ayni = sum(1 for t, i in vk.items() if vb.get(t) == i)
    return {"ortak_onek": n, "ayni_id": ayni, "tam": n == len(mk) and ayni == len(vk)}
