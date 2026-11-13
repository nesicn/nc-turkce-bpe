import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import metinden_egit, zemberek_gerekli

from nc_bpe import cli, egitim, paket, sayim


@pytest.fixture(scope="module")
def derlem_klasoru(belgeler, tmp_path_factory):
    d = tmp_path_factory.mktemp("derlem")
    for i in range(3):
        pq.write_table(pa.table({"metin": belgeler[i::3]}), str(d / f"parca_{i}.parquet"))
    return str(d)


def test_dosya_turleri_ayni_sayim(belgeler, tmp_path):
    from tokenizers import Tokenizer

    from nc_bpe import tarif
    tok_json = json.dumps(tarif.iskelet_json())
    parcalar = belgeler[:200]
    pq.write_table(pa.table({"metin": parcalar}), str(tmp_path / "a.parquet"))
    with open(tmp_path / "b.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"metin": m}, ensure_ascii=False) + "\n" for m in parcalar)
    t = tmp_path / "txt"
    t.mkdir()
    for i, m in enumerate(parcalar):
        (t / f"{i:03d}.txt").write_bytes(m.encode("utf-8"))
    ref = sayim.on_bolme_sayimi(Tokenizer.from_str(tok_json), parcalar)
    for yol in (str(tmp_path / "a.parquet"), str(tmp_path / "b.jsonl"), str(t)):
        say, bilgi = sayim.derlem_sayimi(sayim.dosyalari_bul(yol), tok_json, surec=2)
        assert say == ref and bilgi["belge"] == len(parcalar), yol
    with pytest.raises(FileNotFoundError):
        sayim.dosyalari_bul(str(tmp_path / "yok.csv"))


def test_duz_egitim_metinden_egitimle_ayni(belgeler, derlem_klasoru, tmp_path):
    b = egitim.egit(derlem_klasoru, str(tmp_path / "duz"), 1500, yontem="duz", onbellek=str(tmp_path / "ob"), ilerleme=None)
    paket.hf_paketi_kaydet(metinden_egit(belgeler, 1500), str(tmp_path / "ref"))
    assert b["hatalar"] == [] and b["derlem"]["belge"] == len(belgeler)
    for ad in ("tokenizer.json", "tokenizer_config.json"):
        assert egitim.sha256_dosya(str(tmp_path / "duz" / ad)) == egitim.sha256_dosya(str(tmp_path / "ref" / ad))


@zemberek_gerekli
def test_iki_asamali_uctan_uca(belgeler, derlem_klasoru, tmp_path):
    ob = str(tmp_path / "onbellek")
    b1 = egitim.egit(derlem_klasoru, str(tmp_path / "a"), 1500, 700, zemberek_kelime=500, onbellek=ob, ilerleme=None)
    assert b1["hatalar"] == [] and b1["egitim"]["birinci"] == 700 - 320 and b1["kesin_sinirli_kelime"] > 0
    assert os.listdir(ob) and os.listdir(os.path.join(ob, "zemberek"))
    b2 = egitim.egit(derlem_klasoru, str(tmp_path / "b"), 1500, 700, zemberek_kelime=500, onbellek=ob, ilerleme=None)
    b3 = egitim.egit(derlem_klasoru, str(tmp_path / "c"), 1500, 700, zemberek_kelime=500, onbellek=None, ilerleme=None)
    assert b1["tokenizer_json_sha256"] == b2["tokenizer_json_sha256"] == b3["tokenizer_json_sha256"]
    bilgi = json.load(open(tmp_path / "a" / "egitim_bilgisi.json", encoding="utf-8"))
    assert bilgi["ayarlar"]["v1"] == 700 and bilgi["tokenizer_json_sha256"] == egitim.sha256_dosya(str(tmp_path / "a" / "tokenizer.json"))

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(tmp_path / "a"))
    m = "Ankara'da 1923'te!  \n ışık İĞNE"
    assert hf(m)["input_ids"] == hf(m, add_special_tokens=False)["input_ids"]       # BOS/EOS eklenmez
    assert hf.decode(hf(m)["input_ids"]) == m
    assert (hf.eos_token_id, hf.bos_token_id, hf.pad_token_id, hf.unk_token_id, hf.mask_token_id) == (0, 1, 2, 3, 4)
    assert egitim.yeniden_kur(str(tmp_path / "a" / "tokenizer.json"))["ayni"]


def test_v1_veri_yetmezse_hata(derlem_klasoru, tmp_path, monkeypatch):
    monkeypatch.setattr(egitim, "sinirlar_al", lambda kelimeler, *a, **k: {w: [] for w in kelimeler})
    with pytest.raises(RuntimeError, match="1. aşamada veri yetmedi"):
        egitim.egit(derlem_klasoru, str(tmp_path / "x"), 200_000, 150_000, zemberek_kelime=50, ilerleme=None)


def test_ayar_hatalari(derlem_klasoru, tmp_path):
    for sozluk, v1, yontem in ((1500, None, "iki-asamali"), (300, None, "duz"), (1500, 2000, "iki-asamali"), (1500, 700, "unigram")):
        with pytest.raises(ValueError):
            egitim.egit(derlem_klasoru, str(tmp_path / "x"), sozluk, v1, yontem=yontem, ilerleme=None)


def test_kucuk_derlem_sozluk_hatasi(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("kısa bir metin " * 5, encoding="utf-8")
    with pytest.raises(RuntimeError, match="veri yetersiz"):
        egitim.egit(str(tmp_path / "d"), str(tmp_path / "x"), 4096, yontem="duz", ilerleme=None)


def test_cli_duz_kontrol_ve_yeniden_kur(derlem_klasoru, tmp_path, capsys):
    cikti = str(tmp_path / "duz")
    assert cli.main(["egit", "--derlem", derlem_klasoru, "--cikti", cikti, "--sozluk", "1500", "--yontem", "duz"]) == 0
    assert os.path.isdir(tmp_path / "onbellek")                    # varsayılan önbellek çıktının yanında
    assert cli.main(["kontrol", cikti]) == 0
    assert cli.main(["kontrol", cikti, "--sozluk", "1501"]) == 1
    assert cli.main(["yeniden-kur", os.path.join(cikti, "tokenizer.json")]) == 0
    assert '"ayni": true' in capsys.readouterr().out
