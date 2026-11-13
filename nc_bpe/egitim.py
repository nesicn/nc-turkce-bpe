"""Uçtan uca eğitim: derlem → ön bölme sayımı → (Zemberek sınırları → 1. aşama → 2. aşama) → Hugging Face paketi.

Uzun adımların (sayım, Zemberek) sonuçları `onbellek` klasörüne yazılır; aynı derlem ve aynı ön bölmeyle yeniden
çalıştırıldığında yeniden hesaplanmaz. Bu sayede aynı derlemde farklı sözlük boyutları ve V1 değerleri hızlıca denenir.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import tempfile
import time

from tokenizers import Tokenizer

from . import __version__, bpe, morfoloji, paket, sayim, tarif

ZEMBEREK_KELIME = 1_000_000
_BLOK = 100_000


def sha256_dosya(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def _kimlik(*parcalar) -> str:
    return hashlib.sha256(json.dumps(parcalar, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _pickle_oku(yol: str, kimlik: str):
    if not os.path.exists(yol):
        return None
    with open(yol, "rb") as f:
        kayit = pickle.load(f)
    return kayit["veri"] if kayit.get("kimlik") == kimlik else None


def _pickle_yaz(yol: str, kimlik: str, veri) -> None:
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    gecici = yol + ".tmp"
    with open(gecici, "wb") as f:
        pickle.dump({"kimlik": kimlik, "veri": veri}, f, protocol=5)
    os.replace(gecici, yol)


def _surumler() -> dict:
    import tokenizers
    import transformers
    return {"nc_bpe": __version__, "python": platform.python_version(), "tokenizers": tokenizers.__version__,
            "transformers": transformers.__version__}


# --------------------------------------------------------------------------------------------
# Adımlar
# --------------------------------------------------------------------------------------------
def sayim_al(dosyalar: list[str], iskelet: dict, sutun: str, onbellek: str | None, surec: int, ilerleme=None):
    """Ön bölme sayımı (önbellekten ya da yeniden). Kimlik: dosya adları ve boyutları, metin sütunu, ön bölme."""
    tok_json = json.dumps(iskelet, ensure_ascii=False)
    kim = _kimlik("sayim-v1", [(os.path.basename(d), os.path.getsize(d)) for d in dosyalar], sutun,
                  iskelet["normalizer"], iskelet["pre_tokenizer"])
    yol = os.path.join(onbellek, f"sayim_{kim}.pkl") if onbellek else None
    kayit = _pickle_oku(yol, kim) if yol else None
    if kayit is not None:
        if ilerleme:
            ilerleme(f"Sayım önbellekten: {kayit[1]['parca']:,} parça, {kayit[1]['farkli_parca']:,} farklı")
        return kayit
    if ilerleme:
        ilerleme(f"Sayım: {len(dosyalar)} dosya")
    kayit = sayim.derlem_sayimi(dosyalar, tok_json, sutun, surec, ilerleme)
    if yol:
        _pickle_yaz(yol, kim, kayit)
    return kayit


def sinirlar_al(kelimeler: list[str], onbellek: str | None, surec: int, ilerleme=None) -> dict[str, list[int]]:
    """Kelimelerin kesin morfem sınırları; 100.000 kelimelik bloklar hâlinde önbelleğe yazılır."""
    sinirlar = {}
    for b in range(0, len(kelimeler), _BLOK):
        blok = kelimeler[b:b + _BLOK]
        kim = _kimlik("zemberek-v1", hashlib.sha256("\n".join(blok).encode()).hexdigest())
        yol = os.path.join(onbellek, "zemberek", f"blok_{kim}.pkl") if onbellek else None
        s = _pickle_oku(yol, kim) if yol else None
        if s is None:
            if ilerleme:
                ilerleme(f"Zemberek: {b:,}–{b + len(blok):,} / {len(kelimeler):,}")
            s = morfoloji.kelime_sinirlari(blok, surec=surec, ilerleme=ilerleme)
            if yol:
                _pickle_yaz(yol, kim, s)
        sinirlar.update(s)
    return sinirlar


# --------------------------------------------------------------------------------------------
# Eğitim
# --------------------------------------------------------------------------------------------
YONTEMLER = ("iki-asamali", "duz")


def egit(derlem, cikti: str, sozluk: int, v1: int | None = None, *, yontem: str = "iki-asamali",
         rakam: str = tarif.VARSAYILAN_RAKAM, ekler=tarif.VARSAYILAN_EKLER, sutun: str = "metin",
         zemberek_kelime: int = ZEMBEREK_KELIME, min_frekans: int = tarif.VARSAYILAN_MIN_FREKANS,
         ikinci_pay: float = 0.03, surec: int = 2, onbellek: str | None = None, ilerleme=print) -> dict:
    """Tokenizer'ı eğitir, `cikti` klasörüne tokenizer.json + tokenizer_config.json + egitim_bilgisi.json yazar.

    Döner: egitim_bilgisi.json'un içeriği. Yapısal kontrollerde sorun varsa paket yine yazılır, 'hatalar' doludur.
    """
    if yontem not in YONTEMLER:
        raise ValueError(f"yöntem {YONTEMLER} olmalı, verilen: {yontem!r}")
    if yontem == "iki-asamali" and not v1:
        raise ValueError("iki aşamalı yöntem için V1 gerekli")
    ekler = tarif.ekleri_duzenle(ekler)
    alt = tarif.OZEL_TOKEN_SAYISI + 256
    if sozluk <= alt or (v1 is not None and not alt <= v1 <= sozluk):
        raise ValueError(f"sözlük > {alt} ve {alt} ≤ V1 ≤ sözlük olmalı (sözlük {sozluk}, V1 {v1})")
    t0 = time.time()
    dosyalar = sayim.dosyalari_bul(derlem)
    iskelet = tarif.iskelet_json(rakam, ekler)
    say, sayim_bilgi = sayim_al(dosyalar, iskelet, sutun, onbellek, surec, ilerleme)

    girdi_ist, kesin_sinirli = None, None
    if yontem == "duz":
        if ilerleme:
            ilerleme(f"Düz BPE: sözlük {sozluk:,}")
        j, bilgi = bpe.duz_bpe(say, sozluk, iskelet, min_frekans)
    else:
        kelimeler = morfoloji.en_sik_kelimeler(say, zemberek_kelime)
        sinirlar = sinirlar_al(kelimeler, onbellek, surec, ilerleme)
        kesin_sinirli = sum(1 for s in sinirlar.values() if s)
        girdi, girdi_ist = morfoloji.birinci_asama_girdisi(say, sinirlar)
        del sinirlar
        if ilerleme:
            ilerleme(f"1. aşama: V1 {v1:,} ({len(kelimeler):,} kelime çözümlendi, {kesin_sinirli:,} tanesinde kesin sınır var)")
        m1, ogrenilen = bpe.birinci_asamalar(girdi, [v1], min_frekans)
        del girdi
        if m1[v1] is None:
            raise RuntimeError(f"1. aşamada veri yetmedi: {ogrenilen:,} birleştirme öğrenildi, "
                               f"{bpe.birlestirme_sayisi(v1):,} gerekiyordu. V1'i küçültün ya da derlemi büyütün.")
        j, bilgi = bpe.iki_asamali(say, m1[v1], v1, iskelet, sozluk, min_frekans, ikinci_pay, ilerleme)

    tok = Tokenizer.from_str(json.dumps(j))
    kontrol = paket.yapisal_kontroller(tok, sozluk, rakam, ekler)
    paket.hf_paketi_kaydet(tok, cikti)
    sonuc = {
        "ayarlar": {"yontem": yontem, "sozluk": sozluk, "v1": v1, "rakam": rakam, "ekler": list(ekler), "sutun": sutun,
                    "zemberek_kelime": zemberek_kelime if yontem == "iki-asamali" else None, "min_frekans": min_frekans,
                    "ikinci_pay": ikinci_pay if yontem == "iki-asamali" else None},
        "derlem": {"dosyalar": [os.path.basename(d) for d in dosyalar], **sayim_bilgi},
        "birinci_asama_girdisi": girdi_ist,
        "kesin_sinirli_kelime": kesin_sinirli,
        "egitim": bilgi,
        "tokenizer_json_sha256": sha256_dosya(os.path.join(cikti, "tokenizer.json")),
        "hatalar": kontrol["hatalar"],
        "surumler": _surumler(),
        "sure_dk": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(cikti, "egitim_bilgisi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    if ilerleme:
        durum = "yapısal kontroller temiz" if not kontrol["hatalar"] else "HATA: " + "; ".join(kontrol["hatalar"])
        ilerleme(f"Bitti: {cikti} · sözlük {kontrol['sozluk_boyutu']:,} · {durum} · {sonuc['sure_dk']} dk")
    return sonuc


# --------------------------------------------------------------------------------------------
# Yayımlanmış bir tokenizer'ın yeniden kurulması
# --------------------------------------------------------------------------------------------
def yeniden_kur(tokenizer_json: str) -> dict:
    """Bir tokenizer.json'un birleştirmelerini bu paketin iskeletine yazar, aynı yoldan kaydeder ve dosyaları byte byte
    karşılaştırır. Yanında tokenizer_config.json varsa o da karşılaştırılır. Eğitim yapmaz; iskeletin ve paketlemenin
    yayımlanan dosyayla aynı olduğunu gösterir."""
    with open(tokenizer_json, encoding="utf-8") as f:
        j = json.load(f)
    ayar = tarif.tarif_ayarlari(j)
    if ayar is None:
        return {"ayni": False, "neden": "ön bölme deseni bu tarifin seçeneklerinden hiçbirine uymuyor"}
    rakam, ekler = ayar
    hedef = len(j["model"]["vocab"])
    yeni, _ = bpe.sozluk_kur(tarif.iskelet_json(rakam, ekler), [tuple(m) for m in j["model"]["merges"]], [], hedef)
    sonuc = {"rakam": rakam, "ekler": list(ekler), "sozluk": hedef, "birlestirme": len(j["model"]["merges"])}
    with tempfile.TemporaryDirectory() as d:
        paket.hf_paketi_kaydet(Tokenizer.from_str(json.dumps(yeni)), d)
        klasor = os.path.dirname(os.path.abspath(tokenizer_json))
        for ad in ("tokenizer.json", "tokenizer_config.json"):
            asil = tokenizer_json if ad == "tokenizer.json" else os.path.join(klasor, ad)
            if os.path.exists(asil):
                sonuc[ad] = sha256_dosya(asil) == sha256_dosya(os.path.join(d, ad))
    sonuc["ayni"] = all(v for k, v in sonuc.items() if k.endswith(".json"))
    return sonuc
