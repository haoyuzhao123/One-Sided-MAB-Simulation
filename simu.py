import numpy as np
import matplotlib.pyplot as plt

best = 17

# v is the vector of bidding values, and r is the vector of reserve prices
def revenue(v,r):
    a = 0
    b = 0
    for i in range(len(v)):
        if v[i] > a:
            b = a
            a = v[i]
        elif v[i] > b:
            b = v[i]
    
    res = [0] * len(r)
    for i in range(len(r)):
        if r[i] > a:
            res[i] = 0
        elif r[i] > b:
            res[i] = r[i]
        else:
            res[i] = b
    
    return res

class ELIM:
    def __init__(self,T,K,alpha):
        self.T = T
        self.K = K
        self.alpha = alpha
        a = [0] * self.K
        for i in range(self.K):
            a[i] = i
        self.S = a
        self.mean = [0] * self.K
        self.t = 0
        self.reg = [0]
    def update(self, r):
        t = self.t
        if t == 0:
            rho = 10000
        else:
            rho = np.sqrt(self.alpha * np.log(self.K * self.T * self.T) / ((self.t)))
        a = 0
        for i in range(len(self.S)):
            if self.mean[self.S[i]] > a:
                a = self.mean[self.S[i]]
        lb = a - 2 * rho
        #print('rho = %d, max = %d, S[0] = %d',rho,a,self.mean[self.S[0]])
        tmp = []
        for i in range(len(self.S)):
            if self.mean[self.S[i]] > lb:
                tmp.append(self.S[i])
        self.S = tmp
        for i in range(len(self.S)):
            self.mean[self.S[i]] = float(t) / (t+1) * self.mean[self.S[i]] + 1.0 / (t+1) * r[self.S[i]]
        self.t += 1
        self.reg.append(self.reg[-1] + r[best] - r[self.S[0]])

class EXPRET:
    def __init__(self, T,K):
        self.T = T
        self.K = K
        self.gamma = 1 / np.sqrt(T)
        self.eta = self.gamma / 2
        self.ident = np.zeros(self.K)
        self.ident[0] = 1
        self.p = np.ones(self.K) / self.K
        self.weight = np.ones(self.K)
        self.reg = [0]
    def update(self,r):
        q = (1-self.gamma) * self.p + self.gamma * self.ident
        It = np.random.choice(list(range(self.K)),p=q)
        #print(It)
        #print(q)
        #print(It)
        self.reg.append(self.reg[-1] + r[best] - r[It])
        ident = np.zeros(self.K)
        cum = np.zeros(self.K)
        for i in range(self.K):
            if i >= It:
                ident[i] = 1
            if i == 0:
                cum[i] = q[i]
            else:
                cum[i] = cum[i-1] + q[i]
        l = np.ones(self.K)
        l = l * (1-np.array(r)) / cum * ident
        #print(ident)
        #print(cum)
        #print(l)
        self.weight = self.p * np.exp(-self.eta * l)
        self.p = self.weight / np.sum(self.weight)
        #print(np.sum(self.weight))
        #print(np.sum(self.p))

class UCB:
    def __init__(self, T, K, alpha):
        self.T = T
        self.K = K
        self.alpha = alpha
        self.means = np.zeros(self.K)
        self.times = np.zeros(self.K)
        self.ucb = np.ones(self.K) * 10000
        self.reg = [0]
        self.t = 0
    def update(self, r):
        idx = 0
        maxval = self.ucb[0]
        for i in range(self.K):
            if self.ucb[i] > maxval:
                idx = i
                maxval = self.ucb[i]
        self.reg.append(self.reg[-1] + r[best] - r[idx])
        self.t += 1
        for i in range(self.K):
            if i >= idx:
                self.times[i] += 1
                self.means[i] = (self.times[i]-1) / (self.times[i]) * self.means[i] + r[i] / self.times[i]
            if self.times[i] != 0:
                self.ucb[i] = self.means[i] + np.sqrt(self.alpha * np.log(self.T * self.T * self.K) / (self.times[i]))
            else:
                self.ucb[i] = 10000



def main():
    T = 100000
    K = 20
    delta = 0.05
    mean = np.random.rand(K) * 0.4 + 0.2
    #mean = np.array(list(range(K))) / K * 0.4 + 0.2
    #mean = np.ones(K) * 0.6
    mean[best] = 0.6
    mean[best] += delta
    rp = np.array(list(range(K))) / K
    num_of_times = 100
    elim_total = np.zeros((num_of_times, T+1))
    expret_total = np.zeros((num_of_times, T+1))
    ucb_total = np.zeros((num_of_times, T+1))
    for counter in range(num_of_times):
        if counter % 10 == 0:
            print("simulating %d time." % counter)
        elim = ELIM(T,K,0.5)
        expret = EXPRET(T,K)
        ucb = UCB(T,K,1.5)
        for i in range(T):
            #v = np.random.rand(2)
            #if v[0] > 0.5:
            #    v[0] = 1
            #else:
            #    v[0] = 0
            #if v[1] > 0.5:
            #    v[1] = 1
            #else:
            #    v[1] = 0
            #rp = np.array(list(range(K))) / K
            #print(rp)
            #res = revenue(v,rp)
            #print(res)
            #res = np.random.rand(K) * 0.4 - 0.2 + mean
            #res = np.array(list(range(K))) / K
            #res[-1] = 1
            res = np.random.rand(K)
            for i in range(K):
                if res[i] < mean[i]:
                    res[i] = 1
                else:
                    res[i] = 0
            elim.update(res)
            expret.update(res)
            ucb.update(res)
        elim_total[counter,:] = np.array(elim.reg).reshape((1,T+1))
        expret_total[counter,:] = np.array(expret.reg).reshape((1,T+1))
        ucb_total[counter,:] = np.array(ucb.reg).reshape((1,T+1))

    np.save("elim_total_T100000K20Delta005Best17BernRandmean", elim_total)
    np.save("expret_total_T100000K20Delta005Best17BernRandmean", expret_total)
    np.save("ucb_total_T100000K20Delta005Best17BernRandmean", ucb_total)
    #plt.plot(elim.reg)
    #plt.plot(expret.reg)
    #plt.plot(ucb.reg)
    C = 8 * np.log(K * T * T)
    #elimup = np.ones(T+1)* (C / delta + delta)
    #plt.plot(elimup)
    #plt.legend(['ELIM','EXP3','UCB'])
    #plt.show()
    #print(expret.p)
    #print(elim.S)
    #print(ucb.means)

if __name__ == '__main__':
    main()