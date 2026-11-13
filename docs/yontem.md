# Yöntem

## 1. Neden iki aşama

Standart BPE birleştirmeleri yalnız sıklığa göre seçer. Türkçede sık geçen harf dizileri çoğu zaman kök ile ekin sınırını aşar; böylece sözlüğe `Alb|ümün`, `Dün|yaya` gibi bölmelere yol açan token'lar girer. Morfem öncelikli iki aşamalı eğitim, sözlüğün ilk V1 token'ını morfem sınırı aşılmadan öğrenir. Kalan token'lar bütün derlemde serbestçe öğrenilir: sık kelimelerin bütün hâlleri, sayılar, noktalama, alan terimleri.

Sonuç tek bir sıralı birleştirme listesidir. Çıkarım standart byte düzeyi BPE'dir; özel bir kod, sözlük ya da çalışma zamanı bağımlılığı yoktur.

## 2. Normalizer ve ön bölme

- **NFC.** Küçük harfe çevirme Türkçede kayıplıdır: `I` → `i` olur (`ı` değil), `İ` → `i̇` olur (U+0307 birleşen nokta). NFKC de kayıplıdır: `ﬁ` → `fi`, `x²` → `x2`. NFC, NFC biçimindeki metinde birebir gidiş-dönüş sağlar.
- **Kesme işareti.** GPT-2/GPT-4 desenlerindeki İngilizce kısaltma kuralı (`'s|'t|'d…`) kaldırıldı. Harf ya da rakamdan sonra gelen `'` veya `’` ve ardındaki harfler tek parça olur: `Ankara'da` → `Ankara` · `'da`, `1923'te` → `192` · `3` · `'te`.
- **Kelime.** İsteğe bağlı tek boşluk + harf ve birleşen işaret dizisi. Noktalama kelimeye yapışmaz: `(Ankara)` → `(` · `Ankara` · `)`.
- **Rakamlar üçerli** (`uclu`): bir sayı token'ı en çok üç rakamdır; uzun sayılar 3 rakamlık gruplara bölünür.
- **`sb`:** sayıdan önceki tek boşluk sayıya yapışır (` 2026` → ` 202` · `6`).
- **`ns`:** noktalamanın hemen ardındaki satır sonları noktalamaya yapışır (`.\n\n` tek parça).
- **Satır sonları ve boşluk dizileri** ayrı parçalardır; ön bölme hiçbir karakteri atmaz.

Desen `nc_bpe/tarif.py` içindedir (`on_bolme_deseni`); örnekleri ve beklenen bölmeleri `nc_bpe/paket.py`'deki yapısal kontroller her eğitimden sonra sınar.

## 3. Eğitim adımları

Eğitim metinlerle değil, ön bölme parçalarının sayımıyla yapılır.

1. **Sayım.** Derlemin her belgesi NFC'ye çevrilir ve ön bölmeyle parçalanır. Her byte düzeyi parçanın kaç kez geçtiği sayılır.
2. **Kesin morfem sınırları.** Sayımdaki en sık 1.000.000 kelime türü (yalnız harften oluşan, en az iki harfli) Zemberek'le çözümlenir. Bir kelimenin **kesin sınırları**, bütün çözümlemelerinde ortak olan morfem sınırlarıdır. Kelime çözümlenemezse ya da çözümlemenin yüzeyi kelimenin Türkçe küçük harfli hâlini vermezse kesin sınır yoktur.
3. **1. aşamanın girdisi.**
   - Sınırı hesaplanmış kelimeler kesin sınırlarından bölünür (` kitaplarımızdan` → ` kitap` · `lar` · `ımız` · `dan`); sınırı yoksa bütün kalır.
   - Kelime olmayan parçalar (sayı, noktalama, boşluk, kesme eki) olduğu gibi girer.
   - Sınırı hesaplanmamış seyrek kelimeler karakterlerine bölünür. Böylece yalnız bir karakterin byte'larını birleştirmeye katkı verirler (ör. `Ü`'nün iki byte'ı), karakterler arası birleştirmeye katkı vermezler.
4. **1. aşama.** Bu girdide standart BPE, V1 boyutuna kadar eğitilir. Parça sınırını aşan birleştirme öğrenilemez.
5. **2. aşama.** Derlemin bütün parçaları, bölünmeden, 1. aşamanın birleştirmeleriyle kodlanır. 1. aşamanın her token'ı tek bir özel kullanım alanı (PUA) karakterine eşlenir. BPE bu karakter dizileri üzerinde, sözlüğün kalanını dolduracak kadar eğitilmeye devam eder. Öğrenilen birleştirmeler yeniden byte düzeyi dizgelere çevrilir.
6. **Sözlük.** 64 özel token + 256 byte + 1. aşamanın birleştirmeleri + 2. aşamanın birleştirmeleri, sözlük tam istenen boyuta ulaşana dek sırayla eklenir. Yinelenen bir birleştirme çifti atılır. 2. aşamada sözlükte zaten olan bir dizgeyi üreten birleştirme listede kalır, yeni token eklemez ve `egitim_bilgisi.json`'da `ikinci_cakisma` olarak sayılır.

### PUA eşlemesi

1. aşamanın kimlikleri önce Ek Özel Kullanım Alanı-A'ya (U+F0000–U+FFFFD), o dolunca Ek Özel Kullanım Alanı-B'ye (U+100000–U+10FFFD) eşlenir; toplam 131.068 kimlik sığar. Karakter olmayan kod noktaları (U+FFFFE, U+FFFFF) atlanır. Eşleme kimlik sırasını korur: HF eğiticisi alfabeyi kod noktasına göre sıraladığı için eşit sayılı çiftler, tek seferlik bir eğitimdekiyle aynı sırada seçilir. V1 = 80.000 (128k) B alanını da kullanır; A ve B'de sonuç aynıdır (test: `test_pua_b_ile_ayni_sonuc`).

### Özellikler (testlerle sınanır)

- **V1 = 320** (1. aşama boş) verildiğinde sonuç, aynı sayımla eğitilmiş düz BPE'nin birebir aynısıdır.
- **Sayımdan düz BPE**, metinden doğrudan eğitilen BPE ile byte byte aynıdır.
- **1. aşamanın öneki:** büyük bir V1 ile öğrenilen birleştirmelerin ilk `V1 − 320` tanesi, küçük V1 ile öğrenilenlerin aynısıdır.
- **İç içelik:** aynı V1 ile eğitilen küçük sözlük büyüğün önekidir. Yayımlanan üç tokenizer farklı V1 kullandığı için birbirinin öneki değildir.
- **Determinizm:** aynı derlem ve aynı sürümler aynı `tokenizer.json`'u verir. Zemberek birkaç belirsiz kelimede (yaklaşık 100 binde 3) sonucu Python'un dizgi özeti sırasına göre değiştirir; süreçler `PYTHONHASHSEED=0` ile başlatıldığı için sonuç süreç sayısından bağımsızdır.

## 4. V1'in seçimi

Her sözlük boyutu için birkaç V1 adayı eğitildi ve eğitimde kullanılmamış bir geliştirme kümesinde, ölçümden önce yazılmış bir kabul kuralıyla seçildi. Karşılaştırma ölçütü aynı sayımdan eğitilmiş düz BPE'dir.

Kabul koşulları:

- Kök-ek uyumu F1 ≥ düz BPE + 5 yüzde puanı; kesinlik ve duyarlılık düz BPE'nin altına düşmez.
- 1.000 karakter başına token ≤ düz BPE × 1,01.
- Sözlük tam istenen boyutta; gidiş-dönüş (NFC dahil) %100; yapısal kontroller temiz.

Geçen adaylar arasında F1'i en yüksek olan alınır; F1'i ondan 1 yüzde puanından az geride kalanlar varsa aralarında 1.000 karakter başına token'ı en düşük olan seçilir.

| Sözlük | Adaylar | Seçilen V1 |
|---:|---|---:|
| 32.000 | 8.000 · 16.000 · 24.000 | 24.000 |
| 64.000 | 24.000 · 36.000 · 48.000 | 36.000 |
| 128.000 | 36.000 · 64.000 · 80.000 · 96.000 | 80.000 |

## 5. Paket

`tokenizer.json` + `tokenizer_config.json`:

- `tokenizer_class` = `PreTrainedTokenizerFast`: transformers 4.x ve 5.x'te aynı ID'leri verir. transformers 5.x'in varsayılan olarak yazdığı sınıf 4.x'te yoktur.
- `clean_up_tokenization_spaces` = `False`: `True` olursa decode `" ."` → `"."` yapar ve gidiş-dönüş bozulur.
- `model_max_length` = 131.072. Tokenizer BOS/EOS eklemez: `tokenizer.json`'daki post-processor, kaydederken `transformers` tarafından yazılan etkisiz bir `TemplateProcessing`'tir (özel token eklemez), `add_special_tokens=True` ile `False` aynı ID'leri verir.

## 6. Bilinen sınırlamalar

- Derlemdeki web sayfası kalıpları, büyük harfle bitişik yazılmış birkaç token'ı sözlüğe sokar (ör. `FacebookTwitterPinterest`). 128k sözlükte küçük harften büyük harfe geçiş içeren token'ların tamamı, marka adları dahil, 605'tir (%0,47); bu, böyle token'lar için bir üst sınırdır.
- Zemberek bir yaklaşımdır: çözümleyemediği ya da yanlış çözümlediği kelimelerde 1. aşama kelimeyi bütün bırakır ya da yanlış yerden böler.
- Kesin sınır tanımı temkinlidir: çözümlemeler arasında ortak olmayan sınırlar kullanılmaz.
