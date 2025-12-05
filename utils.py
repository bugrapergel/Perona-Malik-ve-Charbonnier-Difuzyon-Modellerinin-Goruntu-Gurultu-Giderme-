import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from diffusivity_functions import (
    NonlinearDiffusion,
    compute_gradients,
    compute_gradient_magnitude,
)


class ColorNonlinearDiffusion(NonlinearDiffusion):
    """
    Renkli görüntüler için nonlinear diffusion.
    Difüzivite tüm kanalların toplam gradyanından hesaplanır.
    """

    def apply_color(self, color_image):
        image_float = color_image.astype(np.float64)
        channels = [image_float[:, :, i] for i in range(3)]
        result_channels = [ch.copy() for ch in channels]

        history = {
            'mean_r': [], 'mean_g': [], 'mean_b': [],
            'variance_r': [], 'variance_g': [], 'variance_b': [],
            'gradient_magnitude': []
        }

        print(f"Renkli görüntü difüzyonu başlıyor... ({self.diffusivity_type})")

        for iteration in range(self.num_iterations):
            # Her kanal için Gaussian yumuşatma
            if self.sigma > 0:
                smoothed_channels = [
                    gaussian_filter(ch, sigma=self.sigma)
                    for ch in result_channels
                ]
            else:
                smoothed_channels = [ch.copy() for ch in result_channels]

            # Tüm kanalların gradyanları
            gradients = [compute_gradients(ch) for ch in smoothed_channels]

            # Toplam gradyan büyüklüğü
            total_gradient_mag = np.zeros_like(result_channels[0])
            for grad_x, grad_y in gradients:
                grad_mag = compute_gradient_magnitude(grad_x, grad_y)
                total_gradient_mag += grad_mag

            # Ortak difüzivite
            g = self.compute_diffusivity(total_gradient_mag)

            # Her kanalı güncelle
            new_channels = []
            for channel in result_channels:
                grad_x, grad_y = compute_gradients(channel)

                diffusion_x = g * grad_x
                diffusion_y = g * grad_y

                div_x, _ = compute_gradients(diffusion_x)
                _, div_y = compute_gradients(diffusion_y)
                divergence = div_x + div_y

                updated = channel + self.dt * divergence
                new_channels.append(np.clip(updated, 0, 255))

            result_channels = new_channels

            # İstatistikler
            history['mean_r'].append(np.mean(result_channels[0]))
            history['mean_g'].append(np.mean(result_channels[1]))
            history['mean_b'].append(np.mean(result_channels[2]))

            history['variance_r'].append(np.var(result_channels[0]))
            history['variance_g'].append(np.var(result_channels[1]))
            history['variance_b'].append(np.var(result_channels[2]))

            history['gradient_magnitude'].append(np.sum(total_gradient_mag))

            if (iteration + 1) % 10 == 0:
                print(f"  İterasyon {iteration+1}/{self.num_iterations} tamamlandı")

        result = np.stack(result_channels, axis=2)
        return np.clip(result, 0, 255).astype(np.uint8), history


def linear_diffusion(image, num_iterations=50, dt=0.25):
    """
    Linear (Isotropic / Gaussian benzeri) diffusion.
    Karşılaştırma için kullanılır.
    """
    result = image.astype(np.float64)

    for _ in range(num_iterations):
        laplacian = cv2.Laplacian(result, cv2.CV_64F)
        result = result + dt * laplacian
        result = np.clip(result, 0, 255)

    return result.astype(np.uint8)