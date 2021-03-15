from matplotlib import pyplot as plt
import matplotlib, cv2
import numpy as np
from pylorenzmie.analysis import Feature
from pylorenzmie.theory import LMHologram
from pylorenzmie.utilities import coordinates
from CATCH.Estimator_Torch import Estimator

img = cv2.imread('test_image_crop.png')

est = Estimator()
results = est.predict(images = [img])[0]

print(results)

feature = Feature(model=LMHologram(double_precision=False))

# Instrument configuration
ins = feature.model.instrument
ins.wavelength = 0.447     # [um]
ins.magnification = 0.048  # [um/pixel]
ins.n_m = 1.34

# The normalized image constitutes the data for the Feature()
data = img[:,:,0]
data = data / np.mean(data)
feature.data = data

# Specify the coordinates for the pixels in the image data
feature.coordinates = coordinates(data.shape)

p = feature.particle
p.r_p = [data.shape[0]//2, data.shape[1]//2, results['z_p']]
p.a_p = results['a_p']
p.n_p = results['n_p']

holo = feature.hologram()
resid = feature.residuals() +1.

display = np.hstack([data, holo, resid])

matplotlib.use('Qt5Agg')
plt.imshow(display, cmap='gray')
plt.title('Image, Predicted Holo, Residual')
plt.show()
