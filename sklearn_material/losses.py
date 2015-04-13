""" Illustrate loss functions used in regression"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2.5, 2.5, 101)
y = x ** 2
c = 1.345
eps = c / 2
z = .5 * (np.abs(x) < c) * x ** 2 + (np.abs(x) > c) * (c * np.abs(x) - .5 * c ** 2)
u = c * np.abs(np.abs(x) - eps) * (np.abs(x) > eps)

plt.figure(figsize=(5,5))
plt.plot(x, y, linewidth=3, label='squared')
plt.plot(x, z, linewidth=3, color='r', label='huber')
plt.plot(x, u, linewidth=3, color='k', label='eps-insensitive')
plt.plot([0, 0], [0, 6.25],'--k')
plt.legend()
plt.axis('tight')
plt.savefig('losses.png')
plt.show()
