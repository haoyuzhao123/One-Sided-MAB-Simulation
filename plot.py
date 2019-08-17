import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.stats as sps

def confidence_interval(data):
    confidence = 0.99
    n = data.size
    m = np.mean(data)
    std_err = sps.sem(data)
    h = std_err * sps.t.ppf((1 + confidence) / 2, n-1)
    return (m-h, m+h)

def lineplotCI(data, low, upp):
    _, ax = plt.subplots()
    ax.plot(data, alpha = 1,color='b')
    ax.fill_between(low, upp, alpha = 0.8, color='b')

def computeMeanLowUpp(matrix):
    ret_mean = np.mean(matrix, axis = 0)
    num_of_data = matrix.shape[1]
    ret_low = np.zeros(num_of_data)
    ret_upp = np.zeros(num_of_data)
    for t in range(num_of_data):
        ret_low[t], ret_upp[t] = confidence_interval(matrix[:,t])
    return (ret_mean, ret_low, ret_upp)

def plotCI(matrix):
    data, low, upp = computeMeanLowUpp(matrix)
    n = data.size
    #lineplotCI(data, low, upp)
    plt.plot(data)
    plt.fill_between(np.array(list(range(n))), low, upp, alpha = 0.4)
    #plt.plot(low)
    #plt.plot(upp)
    #plt.legend(['data','low','upp'])

def main():
    elim_total = np.load('elim_total_T100000K20Delta005Best17Bern.npy')
    expret_total = np.load('expret_total_T100000K20Delta005Best17Bern.npy')
    ucb_total = np.load('ucb_total_T100000K20Delta005Best17Bern.npy')
    #mean_elim_reg = np.mean(elim_total, axis=0)
    #mean_expret_reg = np.mean(expret_total, axis=0)
    #mean_ucb_reg = np.mean(ucb_total, axis=0)
    #plt.plot(mean_elim_reg)
    #plt.plot(mean_expret_reg)
    #plt.plot(mean_ucb_reg)
    #plt.legend(['ELIM','EXP3','UCB'])
    #plt.show()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.figsize'] = 5,3
    plotCI(expret_total)
    plotCI(elim_total)
    plotCI(ucb_total)
    plt.legend(['EXP3-RTB','ELIM','UCB-N'],loc=2)
    plt.xlabel("Number of rounds")
    plt.ylabel("Regret")
    plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.18)
    plt.show()
    
    
if __name__ == '__main__':
    main()