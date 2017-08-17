import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plot

mean_x = -1
mean_y = 2
sigma_x = 2
sigma_y = 3
covariance = 1
delta = 0.025
x = np.arange(-6.0, 3.0, delta)
y = np.arange(-4.0, 7.0, delta)
X,Y = np.meshgrid(x,y)
Z = mlab.bivariate_normal(X,Y, sigma_x,sigma_y, mean_x, mean_y, covariance)
plot.figure()
contours = plot.contour(X,Y,Z)
plot.clabel(contours, inline=1, fontsize=14)
plot.title('Part (b)')
plot.show()