import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from diffusivity_functions import NonlinearDiffusion


def plot_comparison(original, linear, pm1, pm2, charbonnier, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = [original, linear, pm1, pm2, charbonnier]
    titles = ['Orijinal', 'Linear Diffusion', 'PM Tip 1', 'PM Tip 2', 'Charbonnier']

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3

        if img.ndim == 2:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img)

        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

    axes[1, 2].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_statistics(history, title='İterasyonlar Boyunca İstatistikler', save_path=None):
    # Gri seviye istatistikler
    if 'mean' in history:
        iterations = range(1, len(history['mean']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(iterations, history['mean'], 'b-', linewidth=2)
        axes[0].set_xlabel('İterasyon')
        axes[0].set_ylabel('Ortalama Yoğunluk')
        axes[0].set_title('Ortalama Yoğunluk Değişimi')

        axes[1].plot(iterations, history['variance'], 'r-', linewidth=2)
        axes[1].set_xlabel('İterasyon')
        axes[1].set_ylabel('Varyans')
        axes[1].set_title('Yoğunluk Varyansı Değişimi')

        axes[2].plot(iterations, history['gradient_magnitude'], 'g-', linewidth=2)
        axes[2].set_xlabel('İterasyon')
        axes[2].set_ylabel('Toplam Gradyan Büyüklüğü')
        axes[2].set_title('Toplam Gradyan Büyüklüğü Değişimi')

    # Renkli istatistikler
    elif 'mean_r' in history:
        iterations = range(1, len(history['mean_r']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(iterations, history['mean_r'], 'r-', label='Kırmızı', linewidth=2)
        axes[0].plot(iterations, history['mean_g'], 'g-', label='Yeşil', linewidth=2)
        axes[0].plot(iterations, history['mean_b'], 'b-', label='Mavi', linewidth=2)
        axes[0].set_xlabel('İterasyon')
        axes[0].set_ylabel('Ortalama Yoğunluk')
        axes[0].set_title('Kanal Ortalamaları')
        axes[0].legend()

        axes[1].plot(iterations, history['variance_r'], 'r-', label='Kırmızı', linewidth=2)
        axes[1].plot(iterations, history['variance_g'], 'g-', label='Yeşil', linewidth=2)
        axes[1].plot(iterations, history['variance_b'], 'b-', label='Mavi', linewidth=2)
        axes[1].set_xlabel('İterasyon')
        axes[1].set_ylabel('Varyans')
        axes[1].set_title('Kanal Varyansları')
        axes[1].legend()

        axes[2].plot(iterations, history['gradient_magnitude'], 'k-', linewidth=2)
        axes[2].set_xlabel('İterasyon')
        axes[2].set_ylabel('Toplam Gradyan')
        axes[2].set_title('Toplam Gradyan Büyüklüğü')

    else:
        return

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compare_parameters(image, diff_type='pm1', lambdas=[5, 10, 20],
                       sigmas=[0.5, 1.0, 2.0], save_dir='Results'):
    """
    Lambda ve sigma parametrelerinin etkisini görselleştirir.
    """
    os.makedirs(f'{save_dir}/plots', exist_ok=True)

    # Lambda karşılaştırması
    fig, axes = plt.subplots(1, len(lambdas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')

    for i, lambda_val in enumerate(lambdas, 1):
        diffusion = NonlinearDiffusion(lambda_param=lambda_val, sigma=1.0)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'λ = {lambda_val}')
        axes[i].axis('off')

    plt.suptitle(f'{diff_type.upper()} - Lambda Karşılaştırması',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/lambda_comparison_{diff_type}.png', dpi=300)
    plt.close()

    # Sigma karşılaştırması
    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')

    for i, sigma_val in enumerate(sigmas, 1):
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=sigma_val)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'σ = {sigma_val}')
        axes[i].axis('off')

    plt.suptitle(f'{diff_type.upper()} - Sigma Karşılaştırması',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/sigma_comparison_{diff_type}.png', dpi=300)
    plt.close()