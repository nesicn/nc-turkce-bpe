"""Komut satırı: `python -m nc_bpe <komut>` ya da `nc-bpe <komut>`.

  egit         derlemden tokenizer eğitir
  kontrol      kayıtlı bir tokenizer'ın yapısal kontrollerini çalıştırır
  yeniden-kur  yayımlanmış bir tokenizer.json'u bu paketin iskeletiyle yeniden kurar ve karşılaştırır
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import tarif


def _egit(a) -> int:
    from .egitim import egit
    if a.hazir:
        sozluk, v1 = tarif.HAZIR_AYARLAR[a.hazir]["sozluk"], tarif.HAZIR_AYARLAR[a.hazir]["v1"]
        sozluk, v1 = a.sozluk or sozluk, a.v1 or v1
    else:
        if not a.sozluk:
            print("--sozluk ya da --hazir gerekli", file=sys.stderr)
            return 2
        sozluk, v1 = a.sozluk, a.v1
    if a.yontem == "duz":
        v1 = None
    sonuc = egit(a.derlem, a.cikti, sozluk, v1, yontem=a.yontem, rakam=a.rakam, ekler=a.ekler, sutun=a.sutun,
                 zemberek_kelime=a.zemberek_kelime, min_frekans=a.min_frekans, ikinci_pay=a.ikinci_pay, surec=a.surec,
                 onbellek=a.onbellek or os.path.join(os.path.dirname(os.path.abspath(a.cikti)), "onbellek"))
    return 1 if sonuc["hatalar"] else 0


def _kontrol(a) -> int:
    from tokenizers import Tokenizer

    from .paket import yapisal_kontroller
    yol = a.tokenizer if a.tokenizer.endswith(".json") else os.path.join(a.tokenizer, "tokenizer.json")
    tok = Tokenizer.from_file(yol)
    ayar = tarif.tarif_ayarlari(json.loads(tok.to_str()))
    if ayar is None:
        print("Ön bölme deseni bu tarifin seçeneklerinden hiçbirine uymuyor.", file=sys.stderr)
        return 1
    k = yapisal_kontroller(tok, a.sozluk or tok.get_vocab_size(with_added_tokens=True), *ayar)
    print(json.dumps({"rakam": ayar[0], "ekler": list(ayar[1]), **k}, ensure_ascii=False, indent=2))
    return 1 if k["hatalar"] else 0


def _yeniden_kur(a) -> int:
    from .egitim import yeniden_kur
    s = yeniden_kur(a.tokenizer_json)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0 if s["ayni"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nc-bpe", description="Morfem öncelikli iki aşamalı Türkçe BPE tokenizer.")
    alt = p.add_subparsers(dest="komut", required=True)

    e = alt.add_parser("egit", help="derlemden tokenizer eğitir")
    e.add_argument("--derlem", nargs="+", required=True, help="derlem dosyaları ya da klasörleri (.parquet, .jsonl, .txt)")
    e.add_argument("--cikti", required=True, help="tokenizer'ın yazılacağı klasör")
    e.add_argument("--hazir", choices=sorted(tarif.HAZIR_AYARLAR, key=lambda k: tarif.HAZIR_AYARLAR[k]["sozluk"]),
                   help="yayımlanan tokenizer'ın sözlük boyutu ve V1'i (ör. 64k → 64.000 / 36.000)")
    e.add_argument("--sozluk", type=int, help="toplam sözlük boyutu, özel token'lar dahil")
    e.add_argument("--v1", type=int, help="1. aşamanın sözlük boyutu (özel token'lar ve 256 byte dahil)")
    e.add_argument("--yontem", choices=("iki-asamali", "duz"), default="iki-asamali")
    e.add_argument("--rakam", choices=list(tarif.RAKAM_DESENLERI), default=tarif.VARSAYILAN_RAKAM)
    e.add_argument("--ekler", default=",".join(tarif.VARSAYILAN_EKLER),
                   help=f"ön bölme ekleri, virgülle ({', '.join(tarif.ON_BOLME_EKLERI)}); hiçbiri için boş dizge")
    e.add_argument("--sutun", default="metin", help="parquet sütunu / jsonl alanı")
    e.add_argument("--zemberek-kelime", type=int, default=1_000_000, help="Zemberek'le çözümlenen en sık kelime sayısı")
    e.add_argument("--min-frekans", type=int, default=tarif.VARSAYILAN_MIN_FREKANS)
    e.add_argument("--ikinci-pay", type=float, default=0.03, help="2. aşamada hedefin üstünde öğrenilen birleştirme payı")
    e.add_argument("--surec", type=int, default=2, help="sayım ve Zemberek için süreç sayısı")
    e.add_argument("--onbellek", help="sayım ve Zemberek önbelleği (varsayılan: çıktı klasörünün yanında onbellek/)")
    e.set_defaults(f=_egit)

    k = alt.add_parser("kontrol", help="yapısal kontroller")
    k.add_argument("tokenizer", help="tokenizer.json ya da onu içeren klasör")
    k.add_argument("--sozluk", type=int, help="beklenen sözlük boyutu (varsayılan: dosyadaki)")
    k.set_defaults(f=_kontrol)

    y = alt.add_parser("yeniden-kur", help="yayımlanmış tokenizer.json'u yeniden kurar ve karşılaştırır")
    y.add_argument("tokenizer_json")
    y.set_defaults(f=_yeniden_kur)

    a = p.parse_args(argv)
    return a.f(a)
