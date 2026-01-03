import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMFCN(nn.Module):

    def __init__(self, input_size = 590, lstm_hidden = 128, num_classes = 2):
        super(LSTMFCN, self).__init__()

        # lstm: giriş boyutu = 1 (tek kanal), zaman adımı = input_size
        self.lstm = nn.LSTM(
            input_size = 1, # her zaman adımında 1 özellik
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # CNN: 1D Conv feature extraction (özbitelik çıkarımı)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # tam bağlı katman (lstm + cnn birleşimi)
        self.fc = nn.Linear(lstm_hidden + 64, num_classes)

    def forward(self, x):

        # x boyutu: [batch, features] -> [batch, seq_len, 1]
        x = x.unsqueeze(2) # [B, 590] -> [B, 590, 1]

        # LSTM için transpoze uygula: 
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :] # sadece son adımı al

        # cnn için -> [B, 590, 1] -> [B, 1, 590]
        cnn_input = x.permute(0,2,1)

        # 1D cnn katmanları
        x_cnn = F.relu(self.conv1(cnn_input))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))

        # global average pooling [b, 64, L] -> [B, 64, 1]
        x_cnn = self.gap(x_cnn).squeeze(2) # [B, 64]

        # lstm ve cnn çıktılarını birleştir
        combined = torch.cat([lstm_feat, x_cnn], dim=1) # [B, hidden + 64]

        # sınıflandırı
        out = self.fc(combined) # [B, num_classes] yani hangi sınıfa ait olduğunu return eder

        return out
