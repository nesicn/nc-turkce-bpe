# NC Türkçe BPE

Türkçe için morfem öncelikli, iki aşamalı byte düzeyi BPE tokenizer eğitimi. Yayımlanan 32k / 64k / 128k tokenizer'lar bu kodla eğitildi.

Yöntem standart BPE'ye tek bir kısıt ekler: sözlüğün ilk kısmı, en sık kelimelerin Zemberek ([zemberek-python](https://pypi.org/project/zemberek-python/)) ile bulunan kesin morfem sınırları aşılmadan öğrenilir; geri kalanı bütün derlemde serbestçe öğrenilir. Çıktı sıradan bir `tokenizer.json`'dur (byte düzeyi BPE): `transformers`, `tokenizers` ve Transformers.js ile ek kod olmadan açılır.

## Eğitim

**Colab:** [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nesicn/nc-turkce-bpe/blob/main/notebooks/egitim_colab.ipynb) — kurulum, derlem, eğitim ve kontrol hücre hücre.

**Yerel:**

```bash
pip install "nc-bpe @ git+https://github.com/nesicn/nc-turkce-bpe.git"

# Yayımlanan tokenizer'ların ayarlarıyla
nc-bpe egit --derlem derlem/ --cikti ciktilar/nc-64k --hazir 64k

# Kendi ayarlarınızla
nc-bpe egit --derlem derlem/ --cikti ciktilar/deney-1 --sozluk 64000 --v1 48000

# Karşılaştırma için aynı sayımdan düz BPE
nc-bpe egit --derlem derlem/ --cikti ciktilar/duz-64k --sozluk 64000 --yontem duz
```

Derlem bir ya da daha çok dosya veya klasör olarak verilir: `.parquet` (metin sütunu, varsayılan `metin`), `.jsonl` (metin alanı) ya da `.txt` (her dosya bir belge). Eğitim derlemi bu depoda yoktur; yayımlanan tokenizer'ların derlemi [docs/derlem.md](docs/derlem.md)'de anlatılıyor.

Python'dan:

```python
from nc_bpe.egitim import egit

bilgi = egit("derlem/", "ciktilar/nc-64k", sozluk=64_000, v1=36_000, onbellek="ciktilar/onbellek")
```

Çıktı klasöründe `tokenizer.json`, `tokenizer_config.json` ve `egitim_bilgisi.json` (ayarlar, derlem istatistikleri, iki aşamanın birleştirme sayıları, SHA-256, kütüphane sürümleri) bulunur. Her eğitimden sonra yapısal kontroller çalışır: sözlük tam istenen boyutta mı, özel token ID'leri, Türkçe harflerin tek token olması, ön bölme örnekleri ve gidiş-dönüş. Bir kontrol tutmazsa komut 1 ile çıkar.

Ön bölme sayımı ve Zemberek sınırları, varsayılan olarak çıktı klasörünün yanındaki `onbellek/` klasörüne yazılır. Aynı derlemle yapılan sonraki eğitimler bunları yeniden hesaplamaz; farklı sözlük boyutları ve V1 değerleri bu yüzden hızlı denenir. Derlemin içeriği değişirse önbelleği silin (kimlik dosya adı ve boyutundan hesaplanır).

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--sozluk` | — | Toplam sözlük, özel token'lar dahil |
| `--v1` | — | 1. aşamanın sözlüğü; 320 ≤ V1 ≤ sözlük. 320 verilirse sonuç düz BPE'nin aynısıdır |
| `--yontem` | `iki-asamali` | `duz`: aynı sayımdan standart BPE |
| `--rakam` | `uclu` | `tek` (2·0·2·6), `uclu` (202·6), `hepsi` (2026) |
| `--ekler` | `sb,ns` | Ön bölme ekleri: `sb` sayı boşluğu, `ns` noktalama + satır sonu, `is` birleşen işaretler |
| `--zemberek-kelime` | 1.000.000 | Zemberek'le çözümlenen en sık kelime türü sayısı |
| `--min-frekans` | 2 | İki aşamada da BPE'nin `min_frequency` değeri |
| `--surec` | 2 | Sayım ve Zemberek için süreç sayısı |

Yöntemin ayrıntıları ve V1'in nasıl seçildiği: [docs/yontem.md](docs/yontem.md).

## Yayımlanan tokenizer'lar

| Tokenizer | Sözlük | V1 | `tokenizer.json` SHA-256 |
|---|---:|---:|---|
| [nesicn/nc-turkce-bpe-32k](https://huggingface.co/nesicn/nc-turkce-bpe-32k) | 32.000 | 24.000 | `626f59b8ed602b7b3ee7cf3a07c93f2e6dcc024e57bc24bc5ca68c00ecc9e66e` |
| [nesicn/nc-turkce-bpe-64k](https://huggingface.co/nesicn/nc-turkce-bpe-64k) | 64.000 | 36.000 | `f1b6004d7a279fc11e1eb97b59ada580add684f1f62e629ae7b9938808b5c450` |
| [nesicn/nc-turkce-bpe-128k](https://huggingface.co/nesicn/nc-turkce-bpe-128k) | 128.000 | 80.000 | `9eda551623ff58899a28aed5c8535336c9fda985450dce60c2393dc51efecd23` |

Sözlük boyutları 64 özel token dahildir. V1, 1. aşamanın sözlüğüdür (özel token'lar ve 256 byte dahil). Üçü de aynı derlem, aynı ön bölme sayımı ve aynı Zemberek sınırlarıyla eğitildi; yalnız sözlük boyutu ve V1 değişti.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("nesicn/nc-turkce-bpe-64k")
tok.tokenize("Ankara'da 1923'te Cumhuriyet ilan edildi.")
```

## Tarif

| | |
|---|---|
| Model | Byte düzeyi BPE; `<unk>` hiç üretilmez |
| Normalizer | Yalnız NFC (küçük harfe çevirme ve NFKC yok) |
| Ön bölme | Türkçe kesme işareti: `Ankara'da` → `Ankara` · `'da`; rakamlar üçerli (`2026` → `202` · `6`); sayıdan önceki boşluk sayıya, noktalamadan sonraki satır sonu noktalamaya yapışır |
| Özel token'lar | `<\|endoftext\|>`=0, `<\|startoftext\|>`=1, `<\|pad\|>`=2, `<\|unk\|>`=3, `<\|mask\|>`=4, `<\|im_start\|>`=5, `<\|im_end\|>`=6, `<\|reserved_0\|>`…`<\|reserved_56\|>`=7–63 |
| BOS/EOS | Kendiliğinden eklenmez (post-processor etkisiz) |
| Eğitim | `min_frequency=2`; 1. aşamada en sık 1.000.000 kelime Zemberek'le çözümlenir |

Desen ve kontroller `nc_bpe/tarif.py` ve `nc_bpe/paket.py` içindedir. Python 3.10+ yeter, Java gerekmez (`zemberek-python` saf Python'dur). Sürümler `requirements.txt`'te sabittir; farklı bir `tokenizers` sürümü aynı derlemden farklı bir `tokenizer.json` üretebilir.

## Doğrulama

`yeniden-kur`, yayımlanmış bir `tokenizer.json`'un birleştirmelerini bu paketin iskeletine yazar, aynı yoldan kaydeder ve dosyaları byte byte karşılaştırır. Eğitim yapmaz; normalizer, ön bölme, özel token'lar ve paketlemenin yayımlanan dosyayla aynı olduğunu gösterir.

```bash
hf download nesicn/nc-turkce-bpe-64k tokenizer.json tokenizer_config.json --local-dir yayimlanan/64k
nc-bpe yeniden-kur yayimlanan/64k/tokenizer.json        # "ayni": true
nc-bpe kontrol yayimlanan/64k                           # yapısal kontroller
```

Eğitim deterministiktir: aynı derlem ve aynı kütüphane sürümleri aynı `tokenizer.json`'u verir. Zemberek süreçleri `PYTHONHASHSEED=0` ile başlatılır; BPE'de eşit sayılı çiftler sabit sırayla seçilir.

## Depo

```
nc_bpe/
  tarif.py       özel token'lar, ön bölme deseni, tokenizer iskeleti, hazır ayarlar
  sayim.py       derlemin okunması ve ön bölme sayımı
  morfoloji.py   Zemberek, kesin morfem sınırları, 1. aşamanın girdisi
  bpe.py         sayımdan BPE, düz ve iki aşamalı eğitim, sözlüğün kurulması
  paket.py       Hugging Face paketi ve yapısal kontroller
  egitim.py      uçtan uca eğitim, önbellek, yeniden kurma
  cli.py         komut satırı (nc-bpe)
tests/           pytest; sentetik veriyle gerçek eğitim (`python -m pytest -q`, 42 test, internet gerekmez)
notebooks/       Colab defteri
docs/            yöntem ve derlem
```

## Lisans

Kod [Apache-2.0](LICENSE). Eğitim derlemi bu depoda yoktur; kaynakların lisansları [docs/derlem.md](docs/derlem.md)'de.
