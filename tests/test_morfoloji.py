import collections

from conftest import zemberek_gerekli

from nc_bpe import morfoloji, paket, tarif


def test_parca_kelimesi_ve_parcala(sahte_zemberek):
    tb = tarif.tokenizer_iskeleti()
    parcalar = [p for p, _ in tb.pre_tokenizer.pre_tokenize_str(" kitaplarımızdan Ankara'da 2026, İstanbul")]
    k = [morfoloji.parca_kelimesi(p) for p in parcalar]
    assert k[0] == (1, "kitaplarımızdan") and k[1] == (1, "Ankara") and k[2] is None          # 'da kelime değil
    assert morfoloji.parca_kelimesi(parcalar[-1]) == (1, "İstanbul")
    assert all(morfoloji.parca_kelimesi(p) is None for p in parcalar[3:5])
    zem = sahte_zemberek({"kitaplarımızdan": [["kitap", "lar", "ımız", "dan"]]})
    s = morfoloji.kesin_sinirlar(zem, "kitaplarımızdan")
    assert s == [5, 8, 12]
    p = morfoloji.parcala(parcalar[0], 1, "kitaplarımızdan", s)
    assert [paket.bayttan_metne(x) for x in p] == [" kitap", "lar", "ımız", "dan"] and "".join(p) == parcalar[0]


def test_kesin_sinirlar_ortak_sinir(sahte_zemberek):
    zem = sahte_zemberek({"evlerde": [["ev", "ler", "de"], ["evle", "r", "de"]], "Evde": [["ev", "de"]], "xyz": [["x", "yy"]]})
    assert morfoloji.kesin_sinirlar(zem, "evlerde") == [5]           # yalnız iki çözümlemede ortak sınır
    assert morfoloji.kesin_sinirlar(zem, "Evde") == [2]              # büyük harf: Türkçe küçük harfle karşılaştırılır
    assert morfoloji.kesin_sinirlar(zem, "xyz") == []                # yüzey kelimeyi vermiyor


def test_tr_kucuk():
    assert morfoloji.tr_kucuk("IŞIK İĞNE") == "ışık iğne"


def test_birinci_asama_girdisi_kutle(kurulum):
    _, _, say, _ = kurulum
    sik = morfoloji.en_sik_kelimeler(say, 50)
    sinir = {w: ([2] if len(w) > 3 else []) for w in sik}
    girdi, ist = morfoloji.birinci_asama_girdisi(say, sinir)
    seyrek = sum(c for p, c in say.items() if (k := morfoloji.parca_kelimesi(p)) and k[1] not in sinir)
    assert ist["seyrek_kelime_karakterlerine"] == seyrek
    assert sum(ist.values()) == sum(say.values())
    assert sum(len(p) * c for p, c in girdi.items()) == sum(len(p) * c for p, c in say.items())   # byte'lar korunur
    p = next(p for p in say if (k := morfoloji.parca_kelimesi(p)) and k[1] not in sinir and k[0] == 1 and len(k[1]) > 2)
    bas, w = morfoloji.parca_kelimesi(p)
    assert [paket.bayttan_metne(x) for x in morfoloji.karakterlere(p, bas, w)] == [" "] + list(w)


def test_en_sik_kelimeler_sirasi():
    ters = {b: c for c, b in paket.gpt2_bayt_tablosu().items()}

    def bl(s):
        return "".join(ters[b] for b in s.encode("utf-8"))

    say = collections.Counter({bl(" ev"): 3, bl("ev"): 2, bl(" kapı"): 5, bl(" aa"): 4, bl("1923"): 9, bl(" zz"): 1})
    assert morfoloji.en_sik_kelimeler(say, 3) == ["ev", "kapı", "aa"]       # " ev" ile "ev" aynı kelime; eşitlikte alfabetik


@zemberek_gerekli
def test_kelime_sinirlari_gercek_zemberek_ve_surec_sayisi():
    kelimeler = ["kitaplarımızdan", "evlerde", "gerçekleştirilmiştir", "masa", "Ankara", "qwzx"] * 3
    tek = morfoloji.kelime_sinirlari(kelimeler, surec=1)
    cok = morfoloji.kelime_sinirlari(kelimeler, surec=2, parti=4)
    assert tek == cok and tek["kitaplarımızdan"] == [5, 8, 12] and tek["qwzx"] == []
