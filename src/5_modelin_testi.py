import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import test_loader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns 
import matplotlib.pyplot as plt

# cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modeli oluştur ve ağırlıkları yükle
model = LSTMFCN().to(device)
model.load_state_dict(torch.load("Lstmfcn_secom.pth")) # modelimizin parametrelerini yükle
model.eval() # değerlendime moduna geç

# tahminler ve gerçekler için boş liste hazırla
all_preds = []
all_labels = []

with torch.no_grad(): # test sırasında gradyan hesabına gerek yok
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# skorları hesapla
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Test Doğruluğu:{acc:.4f}")
print(f"F1 Skoru: {f1:.4f}")

print(f"Sınıflandırma raporu:\n{classification_report(all_labels,all_preds)}")

cm = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix:\n{cm}")

plt.figure()
sns.heatmap(cm, annot=True, fmt = "d", cmap = "Blues", xticklabels = ["Normal","Hatalı"], yticklabels = ["Normal","Hatalı"])
plt.xlabel("Tahmin Değeri")
plt.ylabel("Gerçek Değer")
plt.tight_layout()
plt.show()






















