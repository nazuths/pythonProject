import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import json


file = open('data.json',)
data = json.load(file)

prices = [int(x['item_price'][:-1]) for x in data['rows']]

# samples = np.arange(1, len(prices)+1)
# print(samples)

X = np.array(prices).reshape(-1, 1)
print(X)

kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(prices)
X_plot = np.linspace(1,len(prices)+1)
log_dens = kde.score_samples(X_plot.reshape((-1,1)))
print(log_dens)
plt.plot(X_plot, log_dens)
plt.show()