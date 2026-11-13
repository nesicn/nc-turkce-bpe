
import pytest
from tokenizers import Regex
from tokenizers import pre_tokenizers as pt

from nc_bpe import paket, tarif


def bol(metin, rakam="tek", ekler=()):
    p = pt.Split(Regex(tarif.on_bolme_deseni(rakam, ekler)), behavior="isolated", invert=False)
    return [x for x, _ in p.pre_tokenize_str(metin)]


@pytest.mark.parametrize("metin,beklenen", list(paket.ON_BOLME_ORNEKLERI.items()))
def test_on_bolme_turkce_ornekler(metin, beklenen):
    assert bol(metin) == beklenen


def test_ingilizce_kisaltma_kurali_yok():
    for m in ["Ankara'da", "Ankara'dır", "TBMM'si", "İzmir'le", "Ali'm", "Kur'an'ı"]:
        kok, ek = m.split("'", 1)
        assert bol(m) == [kok] + ["'" + p for p in ek.split("'")], bol(m)
    assert bol("1923'te") == ["1", "9", "2", "3", "'te"]


def test_rakam_secenekleri():
    assert bol("2026 yılı") == ["2", "0", "2", "6", " yılı"]
    assert bol("2026 yılı", "uclu") == ["202", "6", " yılı"]
    assert bol("2026 yılı", "hepsi") == ["2026", " yılı"]
    for r in tarif.RAKAM_DESENLERI:
        assert bol("1923'te", r)[-1] == "'te"
    with pytest.raises(ValueError):
        tarif.on_bolme_deseni("ikili")


@pytest.mark.parametrize("ek", tarif.ON_BOLME_EKLERI)
def test_on_bolme_ekleri(ek):
    for metin, ile, ilesiz in paket.EK_ORNEKLERI[ek]:
        assert bol(metin, "uclu", (ek,)) == ile
        assert bol(metin, "uclu", ()) == ilesiz


def test_ekleri_duzenle():
    assert tarif.ekleri_duzenle("ns,sb") == ("sb", "ns") and tarif.ekleri_duzenle("") == ()
    for yanlis in ("sb,sb", "xx"):
        with pytest.raises(ValueError):
            tarif.ekleri_duzenle(yanlis)


def test_on_bolme_kayipsiz():
    for ekler in ((), ("sb", "ns"), tarif.ON_BOLME_EKLERI):
        for m in paket.GIDIS_DONUS_ORNEKLERI:
            assert "".join(bol(m, "uclu", ekler)) == m


def test_ozel_tokenlar():
    assert len(tarif.OZEL_TOKENLAR) == 64 and tarif.OZEL_TOKENLAR[0] == "<|endoftext|>"
    assert all(t.startswith("<|") and t.endswith("|>") for t in tarif.OZEL_TOKENLAR)
    assert tarif.UNK_ID == 3 and set(tarif.TOKEN_ROLLERI.values()) <= set(tarif.OZEL_TOKENLAR)


def test_iskelet():
    j = tarif.iskelet_json()
    v = j["model"]["vocab"]
    assert len(v) == 320 and j["model"]["merges"] == []
    assert all(v[t] == i for i, t in enumerate(tarif.OZEL_TOKENLAR))
    assert [t for t, _ in sorted(v.items(), key=lambda x: x[1])][64:] == sorted(paket.gpt2_bayt_tablosu())
    assert j["normalizer"]["type"] == "NFC" and j["post_processor"] is None and j["decoder"]["type"] == "ByteLevel"
    assert j["model"]["byte_fallback"] is False and j["model"]["unk_token"] is None


def test_tarif_ayarlari_geri_bulunur():
    for rakam in tarif.RAKAM_DESENLERI:
        for ekler in ((), ("sb",), ("sb", "ns"), tarif.ON_BOLME_EKLERI):
            assert tarif.tarif_ayarlari(tarif.iskelet_json(rakam, ekler)) == (rakam, ekler)
    j = tarif.iskelet_json()
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\w+"
    assert tarif.tarif_ayarlari(j) is None


def test_hazir_ayarlar():
    assert {k: (v["sozluk"], v["v1"]) for k, v in tarif.HAZIR_AYARLAR.items()} == \
        {"32k": (32_000, 24_000), "64k": (64_000, 36_000), "128k": (128_000, 80_000)}
