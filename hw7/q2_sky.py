import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plotter
from scipy.misc import imsave

sky = scipy.ndimage.imread('low-rank_data/sky.jpg')
u,s,v = np.linalg.svd(sky, False)
#visualize k values
k_values = [2,4,6,8,10,12,14,16,18,20]
for k in k_values:
    first_k = s[:k]
    new_s = np.append(first_k, np.zeros(len(s) - k))
    new_sky = np.dot(np.dot(u, np.diag(new_s)), v)
    imsave('sky' + str(k) + '.png', new_sky)