"""
LSTM/FCN ile Sensör Hata Tespiti

Problem tanımı: 
    - bir üretim hattındaki sensör verilerine bakarak ürünün hatalı mı yoksa normal mi üretildiğini otomatik olarak tahmin etmek
    - bu sayede hataya hızlı bir şekilde tespit edebilir, bakım maliyetlerini düşürebilir ve kalite kontrol sürecini iyileştirebiliriz

Dataset:
    - UCI SECOM Manufacturing Dataset
    - https://archive.ics.uci.edu/ml/datasets/secom
    - 1567 sample, her biri 590 sensör ölçümünden oluşuyor
    - etiketler:
        - "-1": normal üretim
        - "1" : hatalı üretim
    - özellikler sayısal değerlerden oluşyor, ve bazı sensörler için missing value
    - sorun: veri seti ciddi anlamda dengesi, (%93 normal, %7 hatalı)

Araçlar/Teknolojiler:
    - LSTM-FCN: pytorch
    - imbalanced-learn (smote): veri setini dengelemek için kullan (oversampling)

Plan/Program:
    - 1_load_data.py: veri yükle ve ilk analiz gerçekleştir
    - preprocessing.py: eksik verileri doldur, normalize et ve smote ile dengeleme
    - model.py: LSTM + 1D CNN (FCN) mimarisi oluştur
    - 4_train.py: modeli eğit
    - 5_test.py: modeli test et

install libraries: freeze
pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn tqdm imbalanced-learn

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sensör verilerini yükle
df = pd.read_csv("secom.data", sep = r"\s+", header = None)
print(df.head())

# etiket verisini yükle
labels = pd.read_csv("secom_labels.data", sep = r"\s+", header = None)

# etiket sütununu sensör verisine ekle
df["label"] = labels[0].values

# veri boyutunu yazdır
print(df.shape) # (1567, 591)

print(f"Sınıf dağılımı: \n {df["label"].value_counts()}")

# eksik veri analizi (missing value)

missing_per_column = df.isnull().sum()
total_missing = missing_per_column.sum()
print(f"Toplam eksik değer: {total_missing}")

# eksik oranı hesapla ve büyükten küçüğe doğru sırala
missing_ratio = (missing_per_column/len(df))*100
missing_ratio = missing_ratio[missing_ratio>0].sort_values(ascending=False)
print(missing_ratio)

# eksik verilerin görselleştirilmesi
plt.figure()
sns.barplot(x = missing_ratio.index, y = missing_ratio.values)
plt.title("Eksik veriye sahip sütunlar %")
plt.xlabel("Özellik Index")
plt.ylabel("Eksik Oran (%)")
plt.xticks([])
plt.tight_layout()
plt.show()

# ilk 2 özellik üzerinden basit scatter plot çizimi 
plt.figure()
sns.scatterplot(data = df, x = 0, y = 1, hue = "label", alpha = 0.6, palette="Set1")
plt.xlabel("Özellik 0")
plt.ylabel("Özellik 1")
plt.tight_layout()
plt.show()

# eksik değer oranı fazla olanları atalım
high_missing_cols = missing_ratio[missing_ratio > 85].index
df = df.drop(columns=high_missing_cols)