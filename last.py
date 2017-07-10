import numpy as np
import numpy.random as npr
import scipy.stats as sps
import scipy.special as spsp
import scipy.misc as spm
import scipy.optimize as spo
import numpy.linalg as npl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import sympy as sym
import time
import seaborn as sns
import seaborn.distributions as snsd
import math as math
from label_lines import *

def getLogLhd(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2) - np.log(sigma)

def RWMH(T, mean):
    theta = np.array([mean,np.log(realStd)])
    stepsize = .5/np.sqrt(N)
    S = np.zeros((T, 2))
    acceptance = 0.0
    f = FloatProgress(min=0, max=T)
    display(f)
    for i in range(T):
        accepted = 0
        done = 0
        thetaNew = theta
        thetaP = theta + stepsize*npr.randn(2)
        u = npr.rand()
        ratio = getLogLhd(x, thetaP[0], np.exp(thetaP[1])) - getLogLhd(x, theta[0], np.exp(theta[1]))
        Lambda = np.mean(ratio)
        psi = 1./N*np.log(u)
        if Lambda>psi:
            thetaNew = thetaP
            theta = thetaP
            accepted = 1
            S[i,:] = thetaNew
        else:
            S[i,:] = theta
            
        if i<T/10:
            # Perform some adaptation of the stepsize in the early iterations
            stepsize *= np.exp(1./(i+1)**0.6*(accepted-0.5))
        
        acceptance*=i
        acceptance+=accepted
        acceptance/=(i+1)
        f.value = i
            
    return S

Gradient = lambda x_float, mu_float, sigma_float:np.array([-(2*mu_float - 2*x_float)/(2*sigma_float**2), -1/sigma_float + (-mu_float + x_float)**2/sigma_float**3]).T
Hessian = lambda x_float, mu_float, sigma_float:[[-1/sigma_float**2*np.ones(x_float.shape), 2*(mu_float - x_float)/sigma_float**3], [2*(mu_float - x_float)/sigma_float**3, (1 - 3*(mu_float - x_float)**2/sigma_float**2)/sigma_float**2]]

def langevin(T,mean):
    theta = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    M = N/10. # Size of the subsample
    f = FloatProgress(min=0, max=T)
    display(f) 
    for i in range(T):
        stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        S[i,:] = theta
        f.value = i  
            
    return S


def mala(T,mean):
    theta = np.array([mean, np.log(realStd)])
    thetaP = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    acceptance = 0.0
    M = N/10. # Size of the subsample
    f = FloatProgress(min=0, max=T)
    display(f)
    
    for i in range(T):
        accepted = 0
        stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        #print gradEstimate
        thetaP[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(stepsize)*npr.randn()
        thetaP[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        u = npr.rand()
        ratio = getLogLhd(x, thetaP[0], np.exp(thetaP[1])) - getLogLhd(x, theta[0], np.exp(theta[1]))
        Lambda = np.mean(ratio)
        psi = 1./N*np.log(u)
        if Lambda>psi:
            thetaNew = thetaP
            theta = thetaP
            accepted = 1
            S[i,:] = thetaNew
        else:
            S[i,:] = theta
        
        acceptance*=i
        acceptance+=accepted
        acceptance/=(i+1)
        f.value = i   
            
    return S

def langevin_d(T,mean):
    theta = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    M = N/10. # Size of the subsample
    gamma=0.01
    f = FloatProgress(min=0, max=T)
    display(f)
    for i in range(T):
        #stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        hessianestim = np.sum(Hessian(x[inds], theta[0], np.exp(theta[1])), 0) 
        stepsize = .1/(N*abs(hessianestim[0][i]))/((i+1)**.33)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(2*stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        S[i,:] = theta
        f.value = i
    return S

def langevin_fim(T,mean):
    theta = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    M = N/10. # Size of the subsample
    gamma=0.01
    f = FloatProgress(min=0, max=T)
    display(f)
    for i in range(T):
        #stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        fim = np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0) 
        stepsize = .1/(N*abs(fim[0]))/((i+1)**.33)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(2*stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        S[i,:] = theta
        f.value = i
            
    return S


def langevin_block(T,mean,block):
    return S

def langevin_nesterov(T,mean):
    theta = np.array([mean, np.log(realStd)])
    theta_old = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    M = N/10. # Size of the subsample
    sigma = 0.01
    L1=[]
    L2=[]
    f = FloatProgress(min=0, max=T)
    display(f)
    for i in range(T):
        if i = 1
            stepsize = .1/N/((i+1)**.33)
            inds = npr.randint(0,N,size=M)
            gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
            theta[0] = theta[0] + stepsize*gradEstimate[0] + sigma(theta[0] - theta_old[0]) + np.sqrt(stepsize)*npr.randn()
            theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + sigma(theta[1] - theta_old[1]) + np.sqrt(stepsize)*npr.randn())
            L1.append(theta[0])
            L2.append(theta[1])
            S[i,:] = theta
        else
            stepsize = .1/N/((i+1)**.33)
            inds = npr.randint(0,N,size=M)
            gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
            theta[0] = theta[0] + stepsize*gradEstimate[0] + sigma(theta[0] - L1(i-1)) + np.sqrt(stepsize)*npr.randn()
            theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + sigma(theta[1] - L1(i-1)) + np.sqrt(stepsize)*npr.randn())
            L1.append(theta[0])
            L2.append(theta[1])
            S[i,:] = theta
        f.value = i  
            
    return S

def langevin_block(T,mean):
    theta = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    M = N/10. # Size of the subsample
    f = FloatProgress(min=0, max=T)
    display(f) 
    for i in range(T):
        stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        S[i,:] = theta
        f.value = i  
            
    return S

def langevin_fim(T,mean):
    theta = np.array([mean, np.log(realStd)])
    S = np.zeros((T,2))
    ns = []
    M = N/10. # Size of the subsample
    weights = np.zeros((T,))
    gamma=0.01
    f = FloatProgress(min=0, max=T)
    display(f)
    for i in range(T):
        #stepsize = .1/N/((i+1)**.33)
        inds = npr.randint(0,N,size=M)
        fim = np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        stepsize = .1/(N*abs(fim[0]))/((i+1)**.33)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(2*stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        ns.append(M)
        S[i,:] = theta
        f.value = i    
    return S

# Generate data
npr.seed(1)
N = 100000
dataType = "Gaussian"
#dataType = "logNormal"
x = npr.randn(N)

plt.clf()
plt.hist(x, 30, normed=True)
plt.show()

# We store the mean and std deviation for later reference, they are also the MAP and MLE estimates in this case.
realMean = np.mean(x)
realStd = np.std(x)
print "Mean of x =", realMean
print "Std of x =", realStd


###COMPARISON

niters = 10000
samples_rwmh = [RWMH(niters, mean) for mean in np.arange(0.1, 4, 1)]
samples_ula = [langevin(niters, mean) for mean in np.arange(0.1, 4, 1)]
samples_ulad = [langevin_d(niters, mean) for mean in np.arange(0.1, 4, 1)]
samples_ulanest = [langevin_nesterov(niters, mean) for mean in np.arange(0.1, 4, 1)]
samples_ulablock = [langevin_block(niters, mean) for mean in np.arange(0.1, 4, 1)]
samples_fim = [langevin_fim(niters, mean) for mean in np.arange(0.1, 4, 1)]

###ULA vs RWMH
#several chain
plt.figure(figsize=(20,10))
for samples in samples_rwmh:
    plt.plot(samples[:,0], '-',label=str('MH'))
    labelLines(plt.gca().get_lines(),zorder=2.5)
for samples in samples_ula:
    plt.plot(samples[:,0], '--',label=str('ULA'))
    labelLines(plt.gca().get_lines(),zorder=2.5)
plt.xlim([0, niters])
plt.ylim([-0.1, 5]);


#averaged
rwmh = sum(samples_rwmh)/len(np.arange(0.1, 4, 1))
ula = sum(samples_ula)/len(np.arange(0.1, 4, 1))
plt.figure(figsize=(20,10))
plt.plot(rwmh[:,0], '-',label=str('MH'))
plt.plot(ula[:,0], '--',label=str('ULA'))
labelLines(plt.gca().get_lines(),zorder=2.5)
plt.xlim([0, niters])
plt.ylim([-0.2, 5]);


