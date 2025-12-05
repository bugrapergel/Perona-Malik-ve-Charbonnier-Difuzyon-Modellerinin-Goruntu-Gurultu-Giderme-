import numpy as np
from scipy.ndimage import gaussian_filter


def compute_gradients(image):
    """
    Görüntünün x ve y yönlerindeki gradyanlarını hesaplar (merkezi fark).
    """
    padded = np.pad(image, 1, mode='edge')
    grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    return grad_x, grad_y


def compute_gradient_magnitude(grad_x, grad_y):
    """
    Gradyan büyüklüğünü hesaplar: |∇u| = √(grad_x² + grad_y²)
    """
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


class NonlinearDiffusion:
    """
    Gri tonlamalı görüntüler için Perona-Malik tabanlı nonlinear diffusion.
    """

    def __init__(self, lambda_param=10.0, sigma=1.0, dt=0.25, num_iterations=50):
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.dt = dt
        self.num_iterations = num_iterations
        self.diffusivity_type = 'pm1'  # varsayılan

    def set_diffusivity(self, diff_type):
        """
        'pm1', 'pm2' veya 'charbonnier'
        """
        if diff_type not in ['pm1', 'pm2', 'charbonnier']:
            raise ValueError("Geçersiz difüzivite tipi. 'pm1', 'pm2', veya 'charbonnier' olmalı.")
        self.diffusivity_type = diff_type

    # -------------------------------------------------------------------------
    # Difüzivite fonksiyonları
    # -------------------------------------------------------------------------
    def diffusivity_pm1(self, gradient_magnitude):
        # g(|x|) = exp(-|x|² / λ²)
        return np.exp(-(gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def diffusivity_pm2(self, gradient_magnitude):
        # g(|x|) = 1 / (1 + |x|² / λ²)
        return 1.0 / (1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def diffusivity_charbonnier(self, gradient_magnitude):
        # g(|x|) = 1 / √(1 + |x|² / λ²)
        return 1.0 / np.sqrt(1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def compute_diffusivity(self, gradient_magnitude):
        if self.diffusivity_type == 'pm1':
            return self.diffusivity_pm1(gradient_magnitude)
        elif self.diffusivity_type == 'pm2':
            return self.diffusivity_pm2(gradient_magnitude)
        elif self.diffusivity_type == 'charbonnier':
            return self.diffusivity_charbonnier(gradient_magnitude)

    # -------------------------------------------------------------------------
    # Difüzyon adımı
    # -------------------------------------------------------------------------
    def diffusion_step(self, image):
        """
        Tek bir difüzyon iterasyon adımı.
        PDE: ∂u/∂t = ∇·(g(|∇u_σ|)∇u)
        """
        # Gaussian ile yumuşatma (gradyan için)
        if self.sigma > 0:
            smoothed = gaussian_filter(image, sigma=self.sigma)
        else:
            smoothed = image.copy()

        # Gradyan ve difüzivite
        grad_x, grad_y = compute_gradients(smoothed)
        gradient_mag = compute_gradient_magnitude(grad_x, grad_y)
        g = self.compute_diffusivity(gradient_mag)

        # Orijinal görüntü gradyanları
        grad_x_orig, grad_y_orig = compute_gradients(image)

        # g * ∇u
        diffusion_x = g * grad_x_orig
        diffusion_y = g * grad_y_orig

        # Divergence: ∇·(g∇u)
        div_x, _ = compute_gradients(diffusion_x)
        _, div_y = compute_gradients(diffusion_y)
        divergence = div_x + div_y

        # Explicit adım
        updated_image = image + self.dt * divergence

        return np.clip(updated_image, 0, 255)

    def apply(self, image):
        """
        Nonlinear diffusion uygular, istatistikleri döner.
        """
        result = image.astype(np.float64)
        history = {'mean': [], 'variance': [], 'gradient_magnitude': []}

        for i in range(self.num_iterations):
            result = self.diffusion_step(result)

            history['mean'].append(np.mean(result))
            history['variance'].append(np.var(result))

            gx, gy = compute_gradients(result)
            history['gradient_magnitude'].append(
                np.sum(compute_gradient_magnitude(gx, gy))
            )

            if (i + 1) % 10 == 0:
                print(f"  İterasyon {i+1}/{self.num_iterations} tamamlandı")

        return np.clip(result, 0, 255).astype(np.uint8), history