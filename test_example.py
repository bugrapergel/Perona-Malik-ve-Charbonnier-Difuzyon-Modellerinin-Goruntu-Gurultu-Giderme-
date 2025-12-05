import numpy as np
from PIL import Image
from diffusivity_functions import NonlinearDiffusion

# Görüntüyü yükle (veya test görüntüsü oluştur)
# img = np.array(Image.open('input.png').convert('L'), dtype=np.float64)
img = np.random.randint(0, 255, (256, 256)).astype(np.float64)

# Difüzyon oluştur
diff = NonlinearDiffusion(lambda_param=10.0, sigma=1.0, dt=0.25, num_iterations=50)

# Model seç (pm1, pm2, charbonnier)
diff.set_diffusivity('pm1')

# Uygula
print('Difüzyon uygulanıyor...')
result, history = diff.apply(img)

# Kaydet
Image.fromarray(result).save('../Results/results/output.png')
print('Sonuç kaydedildi: Results/results/output.png')
