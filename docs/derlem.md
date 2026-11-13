# Eğitim derlemi

Yayımlanan üç tokenizer aynı derlemle eğitildi: **768.761 belge, 1.416 MB** ham metin. Derlem, kaynakların lisansları farklı olduğu için bu depoda dağıtılmıyor. Eğitim kodu herhangi bir derlemle çalışır (bkz. README, "Eğitim").

## Kaynaklar

| Kaynak | Veri seti | Lisans | MB | Pay |
|---|---|---|---:|---:|
| Forum | [turkish-nlp-suite/ForumSohbetleri](https://huggingface.co/datasets/turkish-nlp-suite/ForumSohbetleri) | CC BY-SA 4.0 | 490,0 | %32,7 |
| Web | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`tur_Latn`) | ODC-By | 225,0 | %15,0 |
| Haber | [turkish-nlp-suite/Havadis](https://huggingface.co/datasets/turkish-nlp-suite/Havadis) | CC BY-SA 4.0 | 150,0 | %10,0 |
| Ansiklopedi | [turkish-nlp-suite/temiz-Wiki](https://huggingface.co/datasets/turkish-nlp-suite/temiz-Wiki) | CC BY-SA 4.0 | 150,0 | %10,0 |
| Web | [ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) | CC BY 4.0 | 148,4 | %9,9 |
| Akademik | [turkish-nlp-suite/AkademikDerlem](https://huggingface.co/datasets/turkish-nlp-suite/AkademikDerlem) | CC BY-SA 4.0 | 120,0 | %8,0 |
| Sözlük girdileri | [ozertuu/eksiSozlukScrapy](https://huggingface.co/datasets/ozertuu/eksiSozlukScrapy) | Apache-2.0 | 75,8 | %5,1 |
| Blog ve deneme | [turkish-nlp-suite/OzenliDerlem](https://huggingface.co/datasets/turkish-nlp-suite/OzenliDerlem) | CC BY-SA 4.0 | 75,0 | %5,0 |
| Hukuk | [Turkish-NLI/legal_nli_TR_V1](https://huggingface.co/datasets/Turkish-NLI/legal_nli_TR_V1) | Apache-2.0 | 30,0 | %2,0 |
| Film yorumları | [Kaggle: turkish-movie-sentiment-analysis-dataset](https://www.kaggle.com/datasets/mustfkeskin/turkish-movie-sentiment-analysis-dataset) | CC0 (doğrulanmadı) | 27,6 | %1,8 |
| Müşteri yorumları | [turkish-nlp-suite/MusteriYorumlari](https://huggingface.co/datasets/turkish-nlp-suite/MusteriYorumlari) | CC BY-SA 4.0 | 5,2 | %0,3 |
| Tweet | [stefan-it/offenseval2020_tr](https://huggingface.co/datasets/stefan-it/offenseval2020_tr) | CC BY 2.0 | 1,5 | %0,1 |

MB ve paylar, sızıntı denetiminden önceki sürümdendir (790.296 belge, 1.498,4 MB). Sızıntı denetiminde 21.535 belge çıkarıldı; bunların 18.068'i forumdandır.

OzenliDerlem'in Havadis alt kümesi, Havadis veri setinin aynısı olduğu için alınmadı. Hukuk veri seti bazı Türkçe tokenizer değerlendirmelerinde test seti olarak da kullanılıyor; bu tokenizer'ların o setteki sonuçları iyimser olabilir.

## İşlemler

Metne yalnız şu işlemler uygulandı:

1. **Seçim:** doğru metin sütunu; bir forum kaydındaki gönderiler `\n` ile tek belgede; görünür karakteri olmayan kayıt alınmadı; kayıt başına en çok 20.000 karakter (sınırdan önceki son boşlukta kesilir).
2. **Tekrar:** birebir aynı belgeler bir kez alındı.
3. **Etiket öneki:** OffensEval'deki `__label__NOT ` / `__label__OFF ` önekleri silindi.
4. **Cosmos satır sonları:** Cosmos belgelerinin bir kısmında satır sonu iki karakterlik düz metin `\n` olarak yazılmış. Yalnız bu kaynakta `\n` ve `\r\n` gerçek satır sonuna çevrildi; kaçışlanmış `\\n` ve LaTeX komutları (`\neq`, `\nabla`, `\nu`…) değişmedi.
5. **Sızıntı denetimi:** derlem, değerlendirmede kullanılan bütün test setleriyle dört düzeyde karşılaştırıldı: belge, satır (≥ 80 karakter), cümle (≥ 60 karakter) ve yakın kopya (değerlendirme belgesinin kelime 13-gramlarının en az %50'si tek bir eğitim belgesinde). Eşleşen eğitim belgeleri atıldı. Değerlendirmede kullanılan dört kaynakla (tez özetleri, Sözcü 2014 haberleri, MedData, ürün yorumları) örtüşen belgeler de atıldı; bu kaynaklar eğitime alınmadı.

Bozuk kodlama, URL ve HTML varlıkları ölçüldü ama değiştirilmedi.

## Kendi derleminizle

Parquet, JSONL ya da düz metin dosyaları verilebilir. Sayım belge sırasından bağımsızdır; derlemin parçalara nasıl bölündüğü sonucu değiştirmez. Değerlendirmede kullanacağınız metinlerin eğitim derleminde olmadığından emin olun.
