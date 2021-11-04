import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import json

def confidence_interval(data, alpha=0.25):
    srt = sorted(data)
    print(srt[len(srt)//2])
    lower = srt[int((0.5-(1-alpha))*len(srt))]
    upper = srt[int((0.5+(1-alpha))*len(srt))]
    return lower, upper

if __name__ == '__main__':
    file = open('data.json')
    data = json.load(file)

    prices = [int(x['item_price'][:-1]) for x in data['rows']]

    # samples = np.arange(1, len(prices)+1)
    # print(samples)

    X = np.array(prices).reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
    X_plot = np.linspace(min(prices),max(prices))
    log_dens = kde.score_samples(X_plot.reshape((-1,1)))
    plt.plot(X_plot, [10**x for x in log_dens])
    plt.show()

    ci = confidence_interval(prices)
    print(f'Confidence interval: [{ci[0]}, {ci[1]}]')
