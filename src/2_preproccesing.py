import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import TensorDataset, DataLoader

# veriyi yükle
df = pd.read_csv("secom.data", sep = r"\s+", header = None)
labels = pd.read_csv("secom_labels.data", sep = r"\s+", header = None)
df["label"] = labels[0].values

# label sütununa göre ayır
df_normal = df[df["label"] == -1].copy()
df_faulty = df[df["label"] == 1].copy()

# eksik verileri median ile doldur
df_normal = df_normal.fillna(df_normal.median())
df_faulty = df_faulty.fillna(df_faulty.median())

# tekrar birleştir ve df olarak güncelle
df = pd.concat([df_normal, df_faulty], axis=0)

# index sırasını resetle
df = df.reset_index(drop=True)
print(df.shape)
print(df.isnull().sum().sum())  # eksik değer kalmadı mı kontrol

# etiketleri 0-1 olarak dönüştür
df["label"] = df["label"].apply(lambda x: 0 if x == -1 else 1) # 0: normal, 1: hatalı

# X (independent sensör verisi) ve y (hedef değişken) şeklinde ayır
X = df.drop(columns = ["label"]).values
y = df["label"].values

# özellikleri normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE uygulayarak veriyi dengelei hale getir
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# eğitim ve test ayrımı
X_train, X_test, y_train, y_Test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# numpy array den pytorch tensöre dönelim
X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.long)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_test_tensor = torch.tensor(y_Test, dtype = torch.long)

# tensordataset ve dataloader 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
