import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plotter
from scipy.misc import imsave

face = scipy.ndimage.imread('low-rank_data/face.jpg')
u,s,v = np.linalg.svd(face, False)
#visualize k values
k_values = [1,2,3,4,5,6,7,8,9,10]
for k in k_values:
    first_k = s[:k]
    new_s = np.append(first_k, np.zeros(len(s) - k))
    new_face = np.dot(np.dot(u, np.diag(new_s)), v)
    imsave(str(k) + '.png', new_face)
#data_points_x = []
#data_points_y = []
#for k in range(1,101):
#    first_k = s[:k]
#    new_s = np.append(first_k, np.zeros(len(s) - k))
#    new_face = np.dot(np.dot(u, np.diag(new_s)), v)
#    data_points_x.append(k)
#    data_points_y.append(np.linalg.norm(new_face - face, 'fro'))
#data_points_x.append(len(s))
#data_points_y.append(0)
#plotter.plot(data_points_x, data_points_y, 'ro')
#plotter.show()