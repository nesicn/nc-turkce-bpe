import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentetik import derlem  # noqa: E402

from nc_bpe import morfoloji, sayim, tarif  # noqa: E402

zemberek_gerekli = pytest.mark.skipif(not __import__("importlib").util.find_spec("zemberek"), reason="zemberek-python yok")


def metinden_egit(belgeler, sozluk, rakam="uclu", ekler=("sb", "ns")):
    """Metinden doğrudan eğitim (HF BpeTrainer); sayımdan eğitimin karşılaştırılacağı referans."""
    tok = tarif.tokenizer_iskeleti(rakam, ekler)
    tok.train_from_iterator(belgeler, trainer=tarif.egitici(sozluk))
    return tok


@pytest.fixture(scope="session")
def belgeler():
    # Sentetik derlemde büyük Ğ/Ö/Ü çok seyrek; gerçek derlemdeki gibi tek token olmaları için eklenir.
    return derlem(3000, tohum=11) + [f"ĞÖÜ Ğün Ödül Ürün {i}.\n\nYeni" for i in range(300)]


@pytest.fixture(scope="session")
def kurulum(belgeler):
    """(belgeler, iskelet, sayım, sahte sınırlar): en sık 400 kelimede, 6+ harfliler 3. karakterden bölünür."""
    iskelet = tarif.iskelet_json()
    from tokenizers import Tokenizer
    say = sayim.on_bolme_sayimi(Tokenizer.from_str(json.dumps(iskelet)), belgeler)
    sinir = {w: ([3] if len(w) >= 6 else []) for w in morfoloji.en_sik_kelimeler(say, 400)}
    return belgeler, iskelet, say, sinir


class SahteZemberek:
    def __init__(self, d):
        self.d = d

    def cozumle(self, w):
        return self.d.get(w, [])


@pytest.fixture
def sahte_zemberek():
    return SahteZemberek


def kelime_sayimi(say):
    k = collections.Counter()
    for p, c in say.items():
        x = morfoloji.parca_kelimesi(p)
        if x:
            k[x[1]] += c
    return k
