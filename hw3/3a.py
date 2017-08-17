import numpy as np
import matplotlib.pyplot as plot


def mean(tuples):
    mean = [sum(x[0] for x in tuples)/100, sum(y[1] for y in tuples)/100]
    return mean

def plotter(eigenvalues,eigenvectors,x1,x2):
    scale_1 = eigenvalues[0] / pow(pow(eigenvectors[0][0], 2) + pow(eigenvectors[0][1], 2), 0.5)
    scale_2 = eigenvalues[1] / pow(pow(eigenvectors[1][0], 2) + pow(eigenvectors[1][1], 2), 0.5)
    eigen_1_scaled = [x * scale_1 for x in eigenvectors[0]]
    eigen_2_scaled = [x * scale_2 for x in eigenvectors[1]]
    ax = plot.axes()
    bx = plot.axes()
    ax.arrow(0, 0, eigen_1_scaled[0], eigen_1_scaled[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    bx.arrow(0, 0, eigen_2_scaled[0], eigen_2_scaled[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    plot.plot(x1, x2, 'ro')
    plot.axis([-15, 15, -15, 15])
    plot.show()

def rotate(eigenvectors, pairs, x1, x2):
    scale_1 = 1 / pow(pow(eigenvectors[0][0], 2) + pow(eigenvectors[0][1], 2), 0.5)
    scale_2 = 1 / pow(pow(eigenvectors[1][0], 2) + pow(eigenvectors[1][1], 2), 0.5)
    ut = np.array([eigenvectors[0] * scale_1, eigenvectors[1] * scale_2])
    normalized = []
    avg = mean(pairs)
    for x in range(100):
        normalized.append([x1[x] - avg[0], x2[x] - avg[1]])
    for x in range(100):
        normalized[x] = np.dot(ut, normalized[x])
    plot.plot([x[0] for x in normalized], [y[1] for y in normalized], 'ro')
    plot.axis([-15, 15, -15, 15])
    plot.show()
x1 = np.random.normal(3,pow(9, 0.5),100)
x2_raw = np.random.normal(4,pow(4,0.5),100)
x2 = []
pairs = []
for x in range(100):
    x2.append(x1[x]/2 + x2_raw[x])
for x in range(100):
    pairs.append([x1[x],x2[x]])
pairs_array = np.array(pairs).T
covariance = np.cov(pairs_array)
eigenvalues = np.linalg.eig(covariance)[0]
eigenvectors = np.linalg.eig(covariance)[1]
print("mean")
print(mean(pairs))
print("covariance")
print(covariance)
print("eigenvectors")
print(eigenvectors)
print("eigenvalues")
print(eigenvalues)
plotter(eigenvalues,eigenvectors,x1,x2)
rotate(eigenvectors, pairs, x1, x2)