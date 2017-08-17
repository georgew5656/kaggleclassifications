import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plot

mean_x = 1
mean_y = 1
sigma_x = 2
sigma_y = 1
covariance = 0
delta = 0.025
x = np.arange(-3.0, 4.0, delta)
y = np.arange(-4.0, 5.0, delta)
X,Y = np.meshgrid(x,y)
Z1 = mlab.bivariate_normal(X,Y, sigma_x,sigma_y, mean_x, mean_y, covariance)
Z2 = mlab.bivariate_normal(X,Y, 2,2, -1, -1, 1)

plot.figure()
contours = plot.contour(X,Y,Z1-Z2)
plot.clabel(contours, inline=1, fontsize=14)
plot.title('Part (e)')
plot.show()