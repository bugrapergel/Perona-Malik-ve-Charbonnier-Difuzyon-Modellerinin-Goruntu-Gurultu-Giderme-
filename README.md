# Nonlinear Diffusion & Perona-Malik Model Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/Course-CMP717-orange)

Bu depo,  **Doğrusal Olmayan Difüzyon (Anizotropik Difüzyon)** filtreleme tekniklerinin Python implementasyonunu içerir. Proje, görüntüdeki gürültüyü giderirken önemli yapısal kenarları korumayı amaçlayan Perona-Malik modeline odaklanmaktadır.

---

## 👨‍💻 Proje Sahibi
**Buğra PERGEL** Yapay Zeka Mühendisliği, 3. Sınıf  
Ostim Teknik Üniversitesi

---

## 🚀 Özellikler

Bu proje aşağıdaki yeteneklere sahiptir:

* **Çoklu Difüzivite Fonksiyonları:**
    * **Perona-Malik Tip 1:** `exp(-|∇u|²/λ²)` (Güçlü kenar koruma)
    * **Perona-Malik Tip 2:** `1 / (1 + |∇u|²/λ²)` (Geniş aralıklı yumuşatma)
    * **Charbonnier:** `1 / sqrt(1 + |∇u|²/λ²)` (Sayısal olarak kararlı)
* **Renkli Görüntü Desteği:** RGB kanallarının gradyan toplamını (`Joint Diffusivity`) kullanarak renk tutarlılığını koruyan özel implementasyon.
* **Karşılaştırmalı Analiz:** Linear (Gaussian) Difüzyon ile Nonlinear modellerin görsel ve istatistiksel karşılaştırması.
* **Otomatik Sentetik Test:** Harici görsel bulunamazsa, gürültülü sentetik görüntülerle otomatik test yapabilme.

---

## 🧠 Matematiksel Model

Projenin temelinde yatan Kısmi Diferansiyel Denklem (PDE) şudur:

$$\frac{\partial u}{\partial t} = \nabla \cdot (g(|\nabla u_{\sigma}|) \nabla u)$$

Burada:
* $u$: Görüntü yoğunluğu
* $g$: Difüzivite fonksiyonu (Kenarlarda 0'a, düz alanlarda 1'e yaklaşır)
* $\sigma$: Gradyan hesaplaması için Gaussian yumuşatma ölçeği

---

## 📂 Dosya Yapısı

```text
├── code/
│   ├── nonlinear_diffusion.py   # Ana program ve demo akışı
│   ├── diffusivity_functions.py # PM modelleri ve matematiksel formüller
│   ├── utils.py                 # Renkli difüzyon sınıfı ve yardımcılar
│   └── analysis.py              # Grafik çizdirme ve analiz araçları
├── Results/
│   ├── results/                 # İşlenmiş çıktı görüntüleri
│   └── plots/                   # İstatistiksel analiz grafikleri
├── ornek.png                    # Gri test görüntüsü
├── ornekcolor.jpg               # Renkli test görüntüsü
└── README.md                    # Proje dokümantasyonu
