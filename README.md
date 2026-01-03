# LSTM-FCN_MNIST-
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
