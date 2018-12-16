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
sns.set(style="ticks");
plt.ioff() # turn off interactive plotting
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
npr.seed(1)

def plotResults(S, ns, algoName="doesNotMatter", weights="doesNotMatter", figId="basic"):  
    # Plot the joint
    m = np.min(S[:,0])
    M = np.max(S[:,0])
    m_ref = np.min(S_ref[:,0])
    M_ref = np.max(S_ref[:,0])
    
    xlimInf = min(m, m_ref)
    xlimSup = max(M, M_ref)
    xPlot = np.linspace(xlimInf, xlimSup, 1000)
    m = np.min(np.exp(S[:,1]))
    M = np.max(np.exp(S[:,1]))
    m_ref = np.min(np.exp(S_ref[:,1]))
    M_ref = np.max(np.exp(S_ref[:,1]))
    ylimInf = min(m, m_ref)
    ylimSup = max(M, M_ref)
    yPlot = np.linspace(ylimInf, ylimSup, 1000)
    g = sns.jointplot(S[:,0], np.exp(S[:,1]), kind="hex", space=0,size=10, xlim=(xlimInf,xlimSup), ylim=(ylimInf,ylimSup), stat_func=None, marginal_kws={"norm_hist":True}) # 
    plt.sca(g.ax_joint)
    plt.xlabel("$\mu$",)
    plt.ylabel("$\sigma$")

    # Refrence green line
    sns.kdeplot(S_ref[:,0], np.exp(S_ref[:,1]), ax=g.ax_joint, bw="silverman", cmap="BuGn_r", linewidth=5)
    g.ax_marg_x.plot(xPlot, marg0(xPlot), 'g', linewidth=6, label="Ref")
    g.ax_marg_y.plot(marg1(yPlot), yPlot, 'g', linewidth=6)
    
    g.ax_marg_x.legend()
    plt.show()

def getLogLhd(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2) - np.log(sigma)

def RWMH(T):
    theta = np.array([realMean,np.log(realStd)])
    stepsize = .5/np.sqrt(N)
    S = np.zeros((T, 2))
    acceptance = 0.0
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
        if np.mod(i,T/10)==0:
            print "Iteration", i, "Acceptance", acceptance
            
    return S

Gradient = lambda x_float, mu_float, sigma_float:np.array([-(2*mu_float - 2*x_float)/(2*sigma_float**2), -1/sigma_float + (-mu_float + x_float)**2/sigma_float**3]).T
Hessian = lambda x_float, mu_float, sigma_float:[[-1/sigma_float**2*np.ones(x_float.shape), 2*(mu_float - x_float)/sigma_float**3], [2*(mu_float - x_float)/sigma_float**3, (1 - 3*(mu_float - x_float)**2/sigma_float**2)/sigma_float**2]]

def langevin(T):
    theta = np.array([realMean, np.log(realStd)])
    S = np.zeros((T,2))
    ns = []
    M = N/10. # Size of the subsample
    weights = np.zeros((T,))
    
    for i in range(T):
        stepsize = .1/N/((i+1)**.33)
        weights[i] = stepsize
        inds = npr.randint(0,N,size=M)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        theta[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(stepsize)*npr.randn()
        theta[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        ns.append(M)
        S[i,:] = theta
        if np.mod(i,T/10)==0:
            print "Iteration", i
            
    return S, ns, weights 

def mala(T):
    theta = np.array([realMean, np.log(realStd)])
    thetaP = np.array([realMean, np.log(realStd)])
    S = np.zeros((T,2))
    acceptance = 0.0
    ns = []
    M = N/10. # Size of the subsample
    weights = np.zeros((T,))
    
    for i in range(T):
        accepted = 0
        stepsize = .1/N/((i+1)**.33)
        weights[i] = stepsize
        inds = npr.randint(0,N,size=M)
        gradEstimate = N/M*np.sum(Gradient(x[inds], theta[0], np.exp(theta[1])), 0)
        thetaP[0] = theta[0] + stepsize*gradEstimate[0] + np.sqrt(stepsize)*npr.randn()
        thetaP[1] = np.log(np.exp(theta[1]) + stepsize*gradEstimate[1] + np.sqrt(stepsize)*npr.randn())
        ns.append(M)
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
        if np.mod(i,T/10)==0:
            print "Iteration", i, "Acceptance", acceptance
            
    return S, ns, weights 


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

print('\x1b[1;31m'+'Reference'+'\x1b[0m')
#References
S_ref = RWMH(50000)
#Estimate the pdf of my RVS (mean and sd)
marg0 = sps.gaussian_kde(S_ref[:,0])
marg1 = sps.gaussian_kde(np.exp(S_ref[:,1]))

#RWMH
print('\x1b[1;31m'+'RWMH'+'\x1b[0m')
S = RWMH(10000)
plotResults(S, [], algoName="RWMH")

#LANGEVIN
print('\x1b[1;31m'+'Langevin'+'\x1b[0m')
S_L, ns_L, weights_L = langevin(10000)
plotResults(S_L, ns_L, algoName="langevin", weights=weights_L)

#MALA
print('\x1b[1;31m'+'MALA'+'\x1b[0m')
S_M, ns_M, weights_M = mala(10000)
plotResults(S_M, ns_M, algoName="mala", weights=weights_M)