import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from diffusivity_functions import NonlinearDiffusion
from utils import ColorNonlinearDiffusion, linear_diffusion
from analysis import plot_comparison, plot_statistics, compare_parameters


def create_synthetic_image(is_color=False):
    """
    Test için sentetik görüntü üretir (gürültülü + kenarlı).
    Parametre karşılaştırması için kullanıyoruz.
    """
    if is_color:
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img[30:50, :, 0] = 200   # kırmızı şerit
        img[:, 40:60, 1] = 100   # yeşil şerit
        noise = np.random.normal(0, 20, img.shape)
    else:
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img[30:50, :] = 200
        img[:, 40:60] = 50
        noise = np.random.normal(0, 25, img.shape)

    return np.clip(img + noise, 0, 255).astype(np.uint8)


def demo_grayscale():
    print("\nGRİ TONLAMALI GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")

    # Hoca'nın verdiği gri mozaik görüntü
    # Dosya adı: ornek.png  (aynı klasörde olmalı)
    img = cv2.imread('ornek.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            "Gri görüntü okunamadı. 'ornek.png' dosyası bu klasörde mi, adı doğru mu?"
        )

    # Linear difüzyon
    linear_result = linear_diffusion(img, num_iterations=50)

    # Nonlinear difüzyonlar
    results = {}
    for diff_type, name in [('pm1', 'PM Tip 1'),
                            ('pm2', 'PM Tip 2'),
                            ('charbonnier', 'Charbonnier')]:
        print(f"\n{name} difüzyonu uygulanıyor...")
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=1.0,
                                       dt=0.25, num_iterations=50)
        diffusion.set_diffusivity(diff_type)
        result, history = diffusion.apply(img)
        results[diff_type] = (result, history)

        plot_statistics(
            history,
            title=f'{name} - İstatistikler',
            save_path=f'Results/plots/statistics_{diff_type}.png'
        )

    print("\nSonuçlar görselleştiriliyor...")
    plot_comparison(
        img,
        linear_result,
        results['pm1'][0],
        results['pm2'][0],
        results['charbonnier'][0],
        save_path='Results/results/comparison_grayscale.png'
    )


def demo_color():
    print("\nRENKLİ GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")

    # Hoca'nın verdiği renkli mozaik görüntü
    # Dosya adı: ornekcolor.png  (aynı klasörde olmalı)
    img = cv2.imread('ornekcolor.png')
    if img is None:
        raise FileNotFoundError(
            "Renkli görüntü okunamadı. 'ornekcolor.png' dosyası bu klasörde mi, adı doğru mu?"
        )
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Görüntü boyutu: {img.shape}")

    print("\nRenkli PM Tip 1 difüzyonu uygulanıyor...")
    color_diffusion = ColorNonlinearDiffusion(lambda_param=15.0, sigma=1.0,
                                              dt=0.25, num_iterations=30)
    color_diffusion.set_diffusivity('pm1')
    result, history = color_diffusion.apply_color(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Orijinal (Gri)')
    axes[0].axis('off')

    axes[1].imshow(result, cmap='gray')
    axes[1].set_title('PM Tip 1 Difüzyon Sonucu (Gri)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('Results/results/color_diffusion_result.png', dpi=300)
    plt.show()

    plot_statistics(
        history,
        title='Renkli PM Tip 1 - İstatistikler',
        save_path='Results/plots/color_diffusion_statistics.png'
    )


if __name__ == "__main__":
    os.makedirs('Results/results', exist_ok=True)
    os.makedirs('Results/plots', exist_ok=True)

    print("\n" + "=" * 70)
    print("NONLINEAR DIFFUSION - PERONA-MALIK MODEL BAŞLIYOR")
    print("=" * 70)

    print("\nDemo Seçenekleri:")
    print("1. Gri tonlamalı görüntü demo (ornek.png)")
    print("2. Renkli görüntü demo (ornekcolor.png)")
    print("3. Parametre karşılaştırması (sentetik görüntü)")
    print("4. Hepsi")

    choice = input("\nSeçiminiz (1-4): ")

    if choice == '1':
        demo_grayscale()
    elif choice == '2':
        demo_color()
    elif choice == '3':
        synthetic_img = create_synthetic_image(is_color=False)
        compare_parameters(synthetic_img, diff_type='pm1', save_dir='Results')
        compare_parameters(synthetic_img, diff_type='pm2', save_dir='Results')
    elif choice == '4':
        demo_grayscale()
        demo_color()
        synthetic_img = create_synthetic_image(is_color=False)
        compare_parameters(synthetic_img, diff_type='pm1', save_dir='Results')
        compare_parameters(synthetic_img, diff_type='pm2', save_dir='Results')
    else:
        print("Geçersiz seçim!")

    print("\n" + "=" * 70)
    print("PROGRAM TAMAMLANDI")
    print("=" * 70)