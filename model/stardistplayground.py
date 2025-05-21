from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.color import rgb2gray

model = StarDist2D.from_pretrained('2D_versatile_he')

# Load and preprocess image
#img = imread('../A172_Phase_A7_1_00d00h00m_1.tif')
img = imread('../2025-05-20_01-12.png')

if img.ndim == 3:
    print('converting to gray')
    img = rgb2gray(img)
    print(f"img.ndim:{img.ndim}")

img = normalize(img, 1, 99.8)  # use percentiles for H&E images
print(f"img.ndim:{img.ndim}")

# Predict
labels, details = model.predict_instances(img)

# Show result
import matplotlib.pyplot as plt
plt.imshow(labels)
plt.show()