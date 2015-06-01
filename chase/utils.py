import numpy as np
import matplotlib.pyplot as plt

def pfix(p):
    return np.min([np.max([p, 1e-10]), 1-(1e-10)])


def pred_quantiles(p_stop, quantiles=[.25, .5, .75]):
    return np.array([np.sum(np.cumsum(p_stop) < q) for q in quantiles]) + 1


def plot_result(result):
    fig, ax = plt.subplots(figsize=(5, 3))

    for i, label in enumerate(labels):
        ax[i].plot(result_baseline[label]['p_stop_cond'][:,1], styl[0], label='objective', color='black')
        ax[i].plot(result_value[label]['p_stop_cond'][:,1], styl[1], label='utility weighting', color='black')
        ax[i].plot(result_prob[label]['p_stop_cond'][:,1], styl[2], label='probability weighting', color='black')

    for axi in ax:
        axi.legend()
        axi.set_xlim(0, 50)
        axi.set_xlabel('Sample size')

    ax[0].set_ylabel('p(sample size|H)')

    plt.tight_layout()
    plt.show()
