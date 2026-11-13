import json

import pytest
from conftest import metinden_egit
from tokenizers import Tokenizer

from nc_bpe import bpe, morfoloji, paket, tarif


def _model_ayni(a, b):
    return a["model"]["merges"] == b["model"]["merges"] and a["model"]["vocab"] == b["model"]["vocab"]


def _iskelet_ayni(a, b):
    return {k: v for k, v in a.items() if k != "model"} == {k: v for k, v in b.items() if k != "model"}


def test_duz_bpe_metinden_egitimle_ayni(kurulum):
    belgeler, iskelet, say, _ = kurulum
    ref = json.loads(metinden_egit(belgeler, 1500).to_str())
    j, b = bpe.duz_bpe(say, 1500, iskelet)
    assert _model_ayni(j, ref) and _iskelet_ayni(j, ref)
    k, _ = bpe.duz_bpe(say, 1000, iskelet)
    assert bpe.ic_icelik(k, j) == {"ortak_onek": 1000 - 320, "ayni_id": 1000, "tam": True}


def test_bos_birinci_asama_duz_bpe_ile_ayni(kurulum):
    """V1 = 64 + 256 (1. aşama boş) → iki aşamalı yol düz BPE'nin birebir aynısı."""
    _, iskelet, say, _ = kurulum
    duz, _ = bpe.duz_bpe(say, 1500, iskelet)
    j, b = bpe.iki_asamali(say, [], 320, iskelet, 1500)
    assert _model_ayni(j, duz) and b["ikinci_cakisma"] == 0 and b["yinelenen_cift"] == 0


def test_iki_asamali_gecerli_ve_sinir_asmiyor(kurulum):
    belgeler, iskelet, say, sinir = kurulum
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    m1 = bpe.birinci_asamalar(girdi, [700])[0][700]
    parcalar = set(girdi)
    for a, b in m1:                                    # 1. aşamanın her token'ı bir 1. aşama parçasının alt dizgesi
        assert any((a + b) in p for p in parcalar), (a, b)
    j, bilgi = bpe.iki_asamali(say, m1, 700, iskelet, 1500)
    assert j["model"]["merges"][:len(m1)] == [list(m) for m in m1] and bilgi["birinci"] == 700 - 320
    assert _iskelet_ayni(j, iskelet)
    tok = Tokenizer.from_str(json.dumps(j))
    assert not paket.yapisal_kontroller(tok, 1500, "uclu", ("sb", "ns"))["hatalar"]
    hf = paket.hf_tokenizer(tok)
    for m in belgeler[:300] + ["Ankara'da 1923'te ❤️ \r\n\t x² ﬁ", ""]:
        assert hf.decode(hf(m, add_special_tokens=False)["input_ids"]) == m


def test_birinci_asama_onek_ve_veri_yetmezse(kurulum):
    _, _, say, sinir = kurulum
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    hepsi, n = bpe.birinci_asamalar(girdi, (600, 800, 1_000_000))
    assert hepsi[1_000_000] is None and n < bpe.birlestirme_sayisi(1_000_000)
    for v in (600, 800):
        assert hepsi[v] == bpe.birinci_asamalar(girdi, [v])[0][v]            # tek başına eğitimle aynı


def test_ayni_v1_kucuk_sozluk_buyugun_oneki(kurulum):
    _, iskelet, say, sinir = kurulum
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    m1 = bpe.birinci_asamalar(girdi, [700])[0][700]
    kucuk, _ = bpe.iki_asamali(say, m1, 700, iskelet, 1100)
    buyuk, _ = bpe.iki_asamali(say, m1, 700, iskelet, 1500)
    assert bpe.ic_icelik(kucuk, buyuk)["tam"]
    with pytest.raises(ValueError):
        bpe.iki_asamali(say, m1[:-1], 700, iskelet, 1500)


def test_pua_kodu():
    import unicodedata
    assert bpe.pua_kodu(0) == 0xF0000 and bpe.pua_kodu(65_533) == 0xFFFFD
    assert bpe.pua_kodu(65_534) == 0x100000 and bpe.pua_kodu(131_067) == 0x10FFFD
    ks = sorted(set(list(range(0, 131_068, 997)) + [65_533, 65_534, 131_067]))
    assert [bpe.pua_kodu(i) for i in ks] == sorted(bpe.pua_kodu(i) for i in ks)
    for i in (0, 65_533, 65_534, 131_067):
        assert unicodedata.category(chr(bpe.pua_kodu(i))) == "Co"
    with pytest.raises(ValueError):
        bpe.pua_kodu(131_068)
    with pytest.raises(ValueError):
        bpe.pua_karakterleri(131_069)


def test_pua_b_ile_ayni_sonuc(kurulum, monkeypatch):
    """1. aşamanın token'ları PUA-B'ye taştığında (A alanı küçültülerek zorlanır) sonuç birebir aynı. 128k'da
    V1 = 80.000 ile olan durumun küçük ölçekteki karşılığı."""
    _, iskelet, say, sinir = kurulum
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    m1 = bpe.birinci_asamalar(girdi, [700])[0][700]
    j_a, b_a = bpe.iki_asamali(say, m1, 700, iskelet, 1500)
    monkeypatch.setattr(bpe, "PUA_A_BOYUT", 37)
    pua, _ = bpe.ikinci_asama_sayimi(say, m1)
    assert any(ord(ch) >= 0x100000 for k in pua for ch in k)
    j_b, b_b = bpe.iki_asamali(say, m1, 700, iskelet, 1500)
    assert j_b == j_a and b_b == {**b_a, "sure_dk": b_b["sure_dk"]}
    monkeypatch.setattr(bpe, "PUA_A_BOYUT", 0)
    assert bpe.iki_asamali(say, m1, 700, iskelet, 1500)[0] == j_a


def test_sozluk_kur_hatalari(kurulum):
    _, iskelet, say, _ = kurulum
    sinir = {k[1]: [] for p in say if (k := morfoloji.parca_kelimesi(p))}
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    m1 = bpe.birinci_asamalar(girdi, [700])[0][700]
    with pytest.raises(RuntimeError, match="bekleniyordu"):
        bpe.sozluk_kur(iskelet, m1, [], 1200)
    with pytest.raises(RuntimeError, match="sözlükte yok"):
        bpe.sozluk_kur(iskelet, m1, [("yokboyle", "x")], 1200)
    j, b = bpe.sozluk_kur(iskelet, m1, [m1[0]] + [m1[1]], 700)
    assert b["yinelenen_cift"] == 2 and len(j["model"]["vocab"]) == 700 and len(j["model"]["merges"]) == len(m1)


def test_belirlenimci(kurulum):
    _, iskelet, say, sinir = kurulum
    girdi, _ = morfoloji.birinci_asama_girdisi(say, sinir)
    m1 = bpe.birinci_asamalar(girdi, [700])[0][700]
    assert bpe.iki_asamali(say, m1, 700, iskelet, 1500)[0] == bpe.iki_asamali(say, m1, 700, iskelet, 1500)[0]
    assert tarif.iskelet_json() == iskelet
