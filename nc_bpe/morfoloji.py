"""Zemberek ile kesin morfem sınırları ve 1. aşamanın girdisi.

Bir kelimenin kesin sınırları, Zemberek'in bütün çözümlemelerinde ortak olan morfem sınırlarıdır. Kelime çözümlenemezse
ya da çözümlemenin yüzeyi kelimenin Türkçe küçük harfli hâlini vermezse sınır yoktur.
"""
from __future__ import annotations

import collections
import os
import re
import time

from .paket import gpt2_bayt_tablosu

_KELIME = re.compile(r"[^\W\d_]{2,}")      # yalnız harf, en az iki
_TABLO = gpt2_bayt_tablosu()               # byte düzeyi karakter → byte


# --------------------------------------------------------------------------------------------
# Zemberek
# --------------------------------------------------------------------------------------------
def zemberek_yukle():
    """zemberek-python'u yükler. İki küçük uyumluluk köprüsü (davranış değişmez): antlr4 4.8, Python 3.12+'da kaldırılmış
    `typing.io`'yu arıyor; zemberek de veri dosyalarını `pkg_resources.resource_filename` ile buluyor (setuptools'un yeni
    sürümlerinde yok)."""
    import logging
    import sys
    import types
    import typing
    import warnings
    if "typing.io" not in sys.modules:
        m = types.ModuleType("typing.io")
        m.TextIO = typing.TextIO
        sys.modules["typing.io"] = m
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pkg_resources  # noqa: F401
    except ImportError:
        import importlib.util
        m = types.ModuleType("pkg_resources")
        m.resource_filename = lambda paket, yol: os.path.join(os.path.dirname(importlib.util.find_spec(paket).origin), yol)
        sys.modules["pkg_resources"] = m
    logging.getLogger("zemberek").setLevel(logging.ERROR)
    from zemberek import TurkishMorphology
    return TurkishMorphology.create_with_defaults()


def tr_kucuk(s: str) -> str:
    """Türkçe küçük harf (I → ı, İ → i)."""
    return s.replace("I", "ı").replace("İ", "i").lower()


class Zemberek:
    """Önbellekli çözümleyici. cozumle(kelime) → farklı yüzey bölmelerinin listesi (her biri morfem yüzeyleri listesi);
    çözümlenemiyorsa []."""

    def __init__(self, morfoloji=None):
        self._m = morfoloji
        self.onbellek: dict[str, list[list[str]]] = {}

    @property
    def m(self):
        if self._m is None:
            self._m = zemberek_yukle()
        return self._m

    def cozumle(self, w: str) -> list[list[str]]:
        if w not in self.onbellek:
            sonuc = []
            for r in self.m.analyze(w).analysis_results:
                if r.is_unknown():
                    continue
                parca = [md.surface for md in r.morpheme_data_list if md.surface]
                if parca and parca not in sonuc:
                    sonuc.append(parca)
            self.onbellek[w] = sonuc
        return self.onbellek[w]


def _kumulatif(uzunluklar) -> list[int]:
    out = [0]
    for u in uzunluklar:
        out.append(out[-1] + u)
    return out


def kesin_sinirlar(zem, w: str) -> list[int]:
    """Kelimenin bütün Zemberek çözümlemelerinde ortak morfem sınırları (karakter konumu)."""
    kucuk = tr_kucuk(w)
    if len(kucuk) != len(w):
        return []
    bolmeler = [p for p in zem.cozumle(w) if "".join(p) == kucuk]
    if not bolmeler:
        return []
    kumeler = [set(_kumulatif(len(x) for x in p)[1:-1]) for p in bolmeler]
    return sorted(set.intersection(*kumeler))


def _zem_isci_baslat():
    global _ZEM
    _ZEM = Zemberek()


def _zem_isci(kelimeler):
    return [kesin_sinirlar(_ZEM, w) for w in kelimeler]


def kelime_sinirlari(kelimeler: list[str], surec: int = 2, parti: int = 2000, ilerleme=None) -> dict[str, list[int]]:
    """Kelime → kesin sınırlar, ayrı süreçlerde (her süreç kendi Zemberek'ini yükler).

    Tekrarlanabilirlik: Zemberek birkaç belirsiz kelimede (yaklaşık 100 binde 3) sonucu Python'un dizgi özeti (hash)
    sırasına bağlı olarak değiştirir. Süreçler PYTHONHASHSEED=0 ile başlatılır; aynı kelime listesi her çalıştırmada,
    süreç sayısından bağımsız olarak aynı sonucu verir. "spawn" kullanılır: çok iş parçacıklı bir süreçte "fork"
    kilitlenmeye yol açabilir."""
    import multiprocessing as mp
    t0 = time.time()
    sonuc = {}
    partiler = [kelimeler[i:i + parti] for i in range(0, len(kelimeler), parti)]
    eski = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        with mp.get_context("spawn").Pool(max(1, surec), initializer=_zem_isci_baslat) as havuz:
            for j, (p, s) in enumerate(zip(partiler, havuz.imap(_zem_isci, partiler)), 1):
                sonuc.update(zip(p, s))
                if ilerleme and j % 50 == 0:
                    ilerleme(f"  Zemberek: {len(sonuc):,} / {len(kelimeler):,} kelime · {(time.time() - t0) / 60:.1f} dk")
    finally:
        if eski is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = eski
    return sonuc


# --------------------------------------------------------------------------------------------
# Ön bölme parçaları ve 1. aşamanın girdisi
# --------------------------------------------------------------------------------------------
def parca_kelimesi(parca: str) -> tuple[int, str] | None:
    """Byte düzeyi ön bölme parçası bir kelimeyse (isteğe bağlı baştaki tek boşluk + yalnız harf) → (boşluk byte'ı, kelime)."""
    try:
        metin = bytes(_TABLO[c] for c in parca).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None
    bas = 1 if metin.startswith(" ") else 0
    w = metin[bas:]
    return (bas, w) if _KELIME.fullmatch(w) else None


def en_sik_kelimeler(sayim: collections.Counter, n: int) -> list[str]:
    """Ön bölme sayımındaki kelimelerin (baştaki boşluk ayrı tutulmadan) en sık n tanesi; eşitlikte alfabetik."""
    kelimeler = collections.Counter()
    for p, c in sayim.items():
        k = parca_kelimesi(p)
        if k:
            kelimeler[k[1]] += c
    return [w for w, _ in sorted(kelimeler.items(), key=lambda x: (-x[1], x[0]))[:n]]


def parcala(parca: str, bas: int, w: str, sinirlar: list[int]) -> list[str]:
    """Karakter sınırlarını byte düzeyi parçadaki konumlara çevirip böler (byte düzeyi dizgede her byte bir karakter)."""
    if not sinirlar:
        return [parca]
    bayt = _kumulatif(len(ch.encode("utf-8")) for ch in w)
    kes = [0] + [bas + bayt[s] for s in sinirlar] + [len(parca)]
    return [parca[a:b] for a, b in zip(kes, kes[1:])]


def karakterlere(parca: str, bas: int, w: str) -> list[str]:
    """Byte düzeyi kelime parçasını karakterlerine böler; baştaki boşluk byte'ı ayrı parça olur."""
    bayt = _kumulatif(len(ch.encode("utf-8")) for ch in w)
    return ([parca[:bas]] if bas else []) + [parca[bas + a:bas + b] for a, b in zip(bayt, bayt[1:])]


def birinci_asama_girdisi(sayim: collections.Counter, sinirlar: dict[str, list[int]]) -> tuple[collections.Counter, dict]:
    """1. aşamanın girdisi: (parça → sayı).

    - Sınırı hesaplanmış kelimeler kesin morfem sınırlarından bölünür (sınırı yoksa bütün kalır).
    - Kelime olmayan parçalar (sayı, noktalama, boşluk, kesme eki) olduğu gibi girer.
    - Sınırı hesaplanmamış (seyrek) kelimeler karakterlerine bölünür: yalnız bir karakterin byte'larını birleştiren
      birleştirmelere katkı verirler (ör. `Ü` bütün derlemdeki sıklığıyla öğrenilir), karakterler arası birleştirmeye
      katkı vermezler.
    """
    girdi, ist = collections.Counter(), collections.Counter()
    for p, c in sayim.items():
        k = parca_kelimesi(p)
        if k is None:
            girdi[p] += c
            ist["kelime_disi"] += c
            continue
        bas, w = k
        if w not in sinirlar:
            for x in karakterlere(p, bas, w):
                girdi[x] += c
            ist["seyrek_kelime_karakterlerine"] += c
            continue
        parcalar = parcala(p, bas, w, sinirlar[w])
        for x in parcalar:
            girdi[x] += c
        ist["bolunen_kelime" if len(parcalar) > 1 else "butun_kelime"] += c
    return girdi, dict(ist)
