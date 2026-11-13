"""Derlemin okunması ve ön bölme sayımı.

Eğitim, metinlerin kendisiyle değil ön bölme parçalarının sayımıyla yapılır: derlem bir kez okunur, tarifin normalizer'ı ve
ön bölmesiyle parçalanır, her byte düzeyi parçanın kaç kez geçtiği sayılır. Sayım belge sırasından bağımsızdır.

Desteklenen dosyalar (bir klasör verilirse altındaki bütün dosyalar, adlarına göre sıralı):
- .parquet : her satır bir belge; metin sütunu `sutun` (varsayılan "metin")
- .jsonl   : her satır bir JSON nesnesi; metin alanı `sutun`
- .txt     : dosyanın tamamı bir belge
"""
from __future__ import annotations

import collections
import json
import os
import time

UZANTILAR = (".parquet", ".jsonl", ".txt")


def dosyalari_bul(yollar) -> list[str]:
    """Dosya ya da klasör yollarından desteklenen dosyaların sıralı listesi."""
    if isinstance(yollar, str):
        yollar = [yollar]
    out = []
    for y in yollar:
        if os.path.isdir(y):
            for kok, _, adlar in os.walk(y):
                out.extend(os.path.join(kok, a) for a in adlar if a.endswith(UZANTILAR))
        elif y.endswith(UZANTILAR) and os.path.isfile(y):
            out.append(y)
        else:
            raise FileNotFoundError(f"desteklenmeyen ya da bulunamayan yol: {y}")
    out = sorted(set(os.path.abspath(p) for p in out))
    if not out:
        raise FileNotFoundError(f"derlem dosyası bulunamadı ({', '.join(UZANTILAR)}): {yollar}")
    return out


def dosya_metinleri(yol: str, sutun: str = "metin") -> list[str]:
    """Bir derlem dosyasındaki belgeler."""
    if yol.endswith(".parquet"):
        import pyarrow.parquet as pq
        return [m for m in pq.read_table(yol, columns=[sutun]).column(sutun).to_pylist() if m]
    if yol.endswith(".jsonl"):
        with open(yol, encoding="utf-8") as f:
            return [m for m in (json.loads(s).get(sutun) for s in f if s.strip()) if m]
    with open(yol, encoding="utf-8", newline="") as f:
        return [f.read()]


def on_bolme_sayimi(tok, metinler) -> collections.Counter:
    """Tarifin normalizer'ı + ön bölmesiyle byte düzeyi ön bölme parçalarının sayımı."""
    bt = getattr(tok, "backend_tokenizer", tok)
    norm, pre = bt.normalizer, bt.pre_tokenizer
    say = collections.Counter()
    for m in metinler:
        say.update(p for p, _ in pre.pre_tokenize_str(norm.normalize_str(m)))
    return say


def _isci_baslat(tok_json: str, sutun: str):
    global _TOK, _SUTUN
    from tokenizers import Tokenizer
    _TOK, _SUTUN = Tokenizer.from_str(tok_json), sutun


def _isci(yol: str) -> tuple[collections.Counter, int]:
    metinler = dosya_metinleri(yol, _SUTUN)
    return on_bolme_sayimi(_TOK, metinler), len(metinler)


def derlem_sayimi(dosyalar: list[str], tok_json: str, sutun: str = "metin", surec: int = 2,
                  ilerleme=None) -> tuple[collections.Counter, dict]:
    """Bütün dosyalarda ön bölme sayımı; her dosya ayrı süreçte ("spawn"). Döner: (sayım, bilgi)."""
    import multiprocessing as mp
    t0, say, belge = time.time(), collections.Counter(), 0
    with mp.get_context("spawn").Pool(max(1, surec), initializer=_isci_baslat, initargs=(tok_json, sutun)) as havuz:
        for i, (c, n) in enumerate(havuz.imap(_isci, dosyalar), 1):
            say.update(c)
            belge += n
            if ilerleme:
                ilerleme(f"  Sayım: {i}/{len(dosyalar)} dosya · {len(say):,} farklı parça · {(time.time() - t0) / 60:.1f} dk")
    return say, {"dosya": len(dosyalar), "belge": belge, "parca": sum(say.values()), "farkli_parca": len(say),
                 "sure_dk": round((time.time() - t0) / 60, 1)}
