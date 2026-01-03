import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import train_loader, test_loader
from tqdm import tqdm

# cihaz seçimi: GPU varsa kullan yoksa cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modeli oluştur
model = LSTMFCN().to(device)

# kayıp fonksiyonu tanımla: 
criterion = nn.CrossEntropyLoss()

# optimizer: adam
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# eğitim parameterleri
num_epochs = 50

# training döngüsü
for epoch in range(num_epochs):

    model.train()  # modeli training moduna al
    train_loss = 0 # eğitim kaybını sıfır olarak ilklendir
    correct = 0     
    total = 0       

    # eğitim verisi üzerinde bir döngü
    for inputs, labels in tqdm(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        # training (learning)
        optimizer.zero_grad() # gradyanları sıfırla
        outputs = model(inputs) # tahmin yap
        loss = criterion(outputs, labels) # kaybı hesapla
        loss.backward() # geri yayılım
        optimizer.step() # ağrılıkların güncellenmesi 

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # ortalama kayıp değeri
    avg_loss = train_loss / total

    # doğruluk
    accuracy = correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

# modeli kaydet
torch.save(model.state_dict(), "lstmfcn_secom.pth")
print("model kaydedildi") 