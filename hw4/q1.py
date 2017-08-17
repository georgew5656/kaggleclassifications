import numpy as np
"""set up parameters"""
w = np.array([-2,1,0]).T
X = np.array([[0,3,1],[1,3,1],[0,1,1],[1,1,1]])
y = np.array([1,1,0,0]).T
param = .07
logistic = lambda x: 1/(1+np.power(np.e, -x))
s = np.array([logistic(x) for x in np.dot(X, w)])
omega = np.diag(s)
"""calculate coefficients for linear system"""
a = np.dot(np.dot(X.T, omega), X) + 2*param*np.identity(3)
b = np.dot(X.T, y-s) - 2*param*w
"""solve linear system"""
e = np.linalg.solve(a,b)
"""update w,s, and omega"""
w = w + e
s = np.array([logistic(x) for x in np.dot(X, w)])
omega = np.diag(s)
print(w)
print(s)
"""calculate coefficeints for linear system"""
a = np.dot(np.dot(X.T, omega), X) + 2*param*np.identity(3)
b = np.dot(X.T, y-s) - 2*param*w
"""solve linear system"""
e = np.linalg.solve(a,b)
"""update w,s, and omega"""
w = w + e
s = np.array([logistic(x) for x in np.dot(X, w)])
omega = np.diag(s)
print(w)