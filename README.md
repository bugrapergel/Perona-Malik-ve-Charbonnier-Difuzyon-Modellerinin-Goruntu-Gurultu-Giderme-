# Perona-Malik-ve-Charbonnier-Difuzyon-Modellerinin-Goruntu-Gurultu-Giderme-
Python implementation of Nonlinear Diffusion models (Perona-Malik &amp; Charbonnier) for edge-preserving image denoising and smoothing. Includes statistical analysis and color image support
Hazırlayan: Buğra PERGEL
# Bölüm: Yapay Zeka Mühendisliği, 3. Sınıf
# Okul: Ostim Teknik Üniversitesi


1. ÖDEV ÖZETİ
-------------------------------------------------------------------------
Bu proje, Perona-Malik (Tip 1, Tip 2) ve Charbonnier difüzyon modellerini 
kullanarak görüntülerde gürültü giderme ve kenar koruma işlemlerini 
gerçekleştirir. Ayrıca renkli görüntüler için ortak gradyan tabanlı bir 
yaklaşım (Color Nonlinear Diffusion) implemente edilmiştir.

2. DOSYA YAPISI (ZIP İÇERİĞİ)
-------------------------------------------------------------------------
Teslim edilen  dosya aşağıdaki klasör yapısına sahiptir:


├── README.txt                  <-- (Bu dosya: Proje açıklaması)
├── code/                       <-- Python kaynak kodları
│   ├── nonlinear_diffusion.py  <-- Ana çalıştırılabilir dosya (Main)
│   ├── diffusivity_functions.py<-- PM modelleri ve matematiksel formüller
│   ├── utils.py                <-- Renkli difüzyon sınıfı ve yardımcılar
│   └── analysis.py             <-- Grafik çizdirme ve analiz fonksiyonları
└── Results/                    <-- Çıktılar
    ├── rapor.pdf               <-- IEEE formatındaki detaylı proje raporu
    ├── results/                <-- İşlenmiş sonuç görüntüleri (PNG/JPG)
    └── plots/                  <-- İstatistiksel analiz grafikleri

3. KURULUM VE GEREKSİNİMLER
-------------------------------------------------------------------------
Proje Python 3 üzerinde çalışmaktadır. Gerekli kütüphaneler:
- numpy
- opencv-python (cv2)
- matplotlib
- scipy

Kurulum için terminal komutu:
pip install numpy opencv-python matplotlib scipy

4. ÇALIŞTIRMA TALİMATLARI
-------------------------------------------------------------------------
Kodu çalıştırmak için ana dizinde terminali açın ve şu komutu girin:

python code/nonlinear_diffusion.py

NOT: Kod, test için "ornek.png" ve "ornekcolor.png" (veya .jpg) dosyalarını 
ana dizinde arar. Eğer bulamazsa, analizlerin aksamaması için otomatik olarak 
sentetik (yapay) test görüntüleri üretir ve işlemi tamamlar.

5. MENÜ SEÇENEKLERİ
-------------------------------------------------------------------------
Program başladığında aşağıdaki seçenekleri sunar:
[1] Gri Tonlamalı Demo: Mozaik görüntüsü üzerinde PM modellerini karşılaştırır.
[2] Renkli Demo: Renkli mozaik üzerinde renk korumalı difüzyon yapar.
[3] Parametre Analizi: Lambda ve Sigma parametrelerinin etkisini test eder.
[4] Hepsi: Tüm analizleri sırayla yapar ve `Results/` klasörüne kaydeder.

-------------------------------------------------------------------------
