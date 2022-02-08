from __future__ import division, print_function
import numpy as np
from numpy.random import randn
from numpy.fft import rfft
from numpy import asarray
from scipy import signal
from scipy.misc import derivative
from PIL import Image
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

coef = []
coef.append(signal.butter(4, 0.03, analog=False))
coef.append(signal.butter(4, 0.03, analog=False))

sig_ff = []

for i in [1, 2]:
  array = np.transpose(asarray(Image.open("chart" + str(i) + ".png")))
  graph = []

  x = 2 * len(array) / 40
  
  while x / len(array) * 40 <= 38:
    s = 0
    c = 0
    y = 0
    
    while y < len(array[int(x)]):
      if array[int(x)][y] != 0:
        s += y
        c += 1
      
      y += 1
  
    if c:
      graph.append(- ((s / c) / len(array[int(x)]) * 225 - 100))
    else:
      graph.append(0)
      
    x += 3 / 32 * len(array) / 40
  
  sig_ff.append(signal.filtfilt(coef[i - 1][0], coef[i - 1][1], np.array(graph)))

sig_diff = sig_ff[0] - sig_ff[1]
sig_grad = np.gradient(sig_ff[1])
sig_grad2 = np.gradient(sig_grad)

plt.subplot(2, 1, 1)
plt.title("Micro-Gal level gravity measurements with cold atom interferometry")
plt.grid(True, which='both')
plt.plot(sig_ff[0], label='Experimental Data')
plt.plot(sig_ff[1], label='Tide Model')
plt.legend(loc="upper center")

plt.subplot(2, 1, 2)
plt.title("Residual of Micro-Gal level gravity measurements with cold atom interferometry")
plt.grid(True, which='both')
plt.plot(sig_diff / np.amax(sig_diff), label='Normalized Residual')
plt.plot(sig_grad / np.amax(sig_grad), label='Normalized Derivative')
plt.legend(loc="upper center")

mng = plt.get_current_fig_manager()
mng.resize(1024, 768)

plt.show()

