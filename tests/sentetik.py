"""Testler için sabit tohumlu, Türkçe biçimli sentetik metin üreteci.

Gerçek veri değildir; yalnız kodun doğru çalıştığını sınamak içindir. Metrik değerleri gerçek Türkçe hakkında bilgi vermez.
"""
from __future__ import annotations

import random

KOKLER = ("ev okul kitap göz yol iş gün su el yüz baş dil söz kız çocuk araba masa kapı pencere şehir "
          "ülke deniz dağ ağaç çiçek güneş ay yıldız hava toprak ateş taş demir altın para zaman yıl "
          "hafta sabah akşam gece öğretmen öğrenci doktor hasta hastane ilaç bilgi bilgisayar telefon "
          "haber gazete dergi film müzik şarkı oyun spor futbol takım maç gol yarış koşu yüzme "
          "gel git yap et al ver bak gör bil düşün söyle konuş yaz oku çalış otur kalk yürü koş "
          "sev iste anla başla bitir aç kapa getir götür bekle sor cevap ver bul kaybet kazan").split()
EKLER_KALIN = ["lar", "ı", "a", "da", "dan", "ın", "ımız", "ları", "daki", "lık", "cı", "sız", "la"]
EKLER_INCE = ["ler", "i", "e", "de", "den", "in", "imiz", "leri", "deki", "lik", "ci", "siz", "le"]
FIIL_EKLERI = ["mak", "mek", "ıyor", "iyor", "dı", "di", "acak", "ecek", "mış", "miş", "abilir", "ebilir"]
OZEL_ISIMLER = ["Ankara", "İstanbul", "İzmir", "Türkiye", "Atatürk", "TBMM", "ABD", "Avrupa", "Iğdır",
                "Çanakkale", "Şanlıurfa", "Ümraniye", "Özlem", "Ağrı", "NASA", "Google"]
KESME_EKLERI = ["'da", "'de", "'nın", "'nin", "'ya", "'ye", "'dan", "'den", "'ta", "'te", "'yı", "'yi",
                "’da", "’nin", "'dır", "'dir"]
NOKTALAMA = [".", ",", "!", "?", ":", ";", "...", "…", "\"", "(", ")", "-", "—"]
HECELER = [a + b for a in "bcçdfgğhjklmnprsştvyz" for b in "aeıioöuü"] + \
          [a + b + c for a in "bcdgkmnpst" for b in "aeıiou" for c in "lmnrsşkt"]
KALIN = set("aıou")


def _uyumlu_ek(kok: str, rnd: random.Random) -> str:
    sesli = [c for c in kok if c in "aeıioöuü"]
    kalin = bool(sesli) and sesli[-1] in KALIN
    return rnd.choice(EKLER_KALIN if kalin else EKLER_INCE)


def kelime(rnd: random.Random) -> str:
    r = rnd.random()
    if r < 0.55:
        k = rnd.choice(KOKLER)
        for _ in range(rnd.choice([0, 0, 1, 1, 2, 3])):
            k += _uyumlu_ek(k, rnd)
        return k
    if r < 0.62:
        return rnd.choice(KOKLER) + rnd.choice(FIIL_EKLERI)
    if r < 0.72:
        return rnd.choice(OZEL_ISIMLER) + (rnd.choice(KESME_EKLERI) if rnd.random() < 0.6 else "")
    if r < 0.78:
        return str(rnd.choice([1923, 2026, 15, 3, 100, 1250, 7])) + (rnd.choice(["'te", "'de", "'da"]) if rnd.random() < 0.3 else "")
    # nadir, rastgele heceli kelimeler: sözlüğün dolabilmesi için çeşitlilik
    w = "".join(rnd.choice(HECELER) for _ in range(rnd.randint(1, 4)))
    return w.capitalize() if rnd.random() < 0.1 else (w.upper() if rnd.random() < 0.02 else w)


def belge(rnd: random.Random, kelime_sayisi: int) -> str:
    parcalar = []
    for i in range(kelime_sayisi):
        w = kelime(rnd)
        if i == 0 or (parcalar and parcalar[-1] in ".!?"):
            w = w[:1].upper() + w[1:] if w[:1] not in "iı" else ("İ" if w[0] == "i" else "I") + w[1:]
        parcalar.append(w)
        r = rnd.random()
        if r < 0.08:
            parcalar.append(rnd.choice(NOKTALAMA))
        if r < 0.015:
            parcalar.append("\n" if rnd.random() < 0.7 else "\n\n")
    metin = ""
    for p in parcalar:
        if not metin or metin.endswith("\n") or p in ".,!?:;…)" or p.startswith("\n"):
            metin += p
        else:
            metin += " " + p
    ekstra = rnd.random()
    if ekstra < 0.03:
        metin += "\n```python\ndef topla(a, b):\n    return a + b\n```\n"
    elif ekstra < 0.06:
        metin += " https://www.ornek.com.tr/haber?id=" + str(rnd.randint(1, 99999))
    elif ekstra < 0.08:
        metin += " 😂👍 çok iyiii!!!"
    return metin


def derlem(belge_sayisi: int, tohum: int = 42, ort_kelime: int = 60) -> list[str]:
    rnd = random.Random(tohum)
    return [belge(rnd, max(3, int(rnd.expovariate(1 / ort_kelime)))) for _ in range(belge_sayisi)]
