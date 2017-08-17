import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plot

mean_x = 1
mean_y = 1
sigma_x = 1
sigma_y = 2
covariance = 0
delta = 0.025
x = np.arange(-1.5, 3.0, delta)
y = np.arange(-3.0, 4.0, delta)
X,Y = np.meshgrid(x,y)
Z = mlab.bivariate_normal(X,Y, sigma_x,sigma_y, mean_x, mean_y, covariance)
plot.figure()
contours = plot.contour(X,Y,Z)
plot.clabel(contours, inline=1, fontsize=14)
plot.title('Part (a)')
plot.show()

