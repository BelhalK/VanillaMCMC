from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
np.random.seed(1234)


### The three algorithms
def mh_chain(niters, theta, target, sigma):
    samples = [theta]
    while len(samples) < niters:
        theta_p = theta + st.norm(0, sigma).rvs()
        rho = min(1, target.pdf(theta_p)/target.pdf(theta))
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
            samples.append(theta)
    return samples

def mala_chain(niters, theta, target, sigma):
    samples = [theta]
    gamma = 0.01
    delta = 0.01
    while len(samples) < niters:
        theta_1 = theta
        theta_2 = theta_1 + delta
        gradU = -(target.pdf(theta_2) - target.pdf(theta_1))/delta
        theta_p = theta +gamma*gradU + st.norm(0, sigma).rvs()
        rho = min(1, target.pdf(theta_p)*st.norm(0, sigma).pdf(theta)/[st.norm(0, sigma).pdf(theta_p)*target.pdf(theta)])
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
            samples.append(theta)
    return samples


def ula_chain(niters, theta, target, sigma):
    samples = [theta]
    gamma = 0.01
    delta = 0.01
    while len(samples) < niters:
        theta_1 = theta
        theta_2 = theta_1 + delta
        gradU = -(target.pdf(theta_2) - target.pdf(theta_1))/delta
        theta_p = theta +gamma*gradU + st.norm(0, sigma).rvs()
        theta = theta_p
        samples.append(theta)
    return samples


def nesterov_chain(niters, theta, target, sigma):
    samples = [theta]
    gamma = 0.01
    delta = 0.01
    R = 0.01
    while len(samples) < niters:
        theta_1 = theta
        theta_2 = theta_1 + delta
        gradU = -(target.pdf(theta_2) - target.pdf(theta_1))/delta
        theta_p = theta +gamma*gradU + R*(theta - theta[k-1]) + st.norm(0, sigma).rvs()
        rho = min(1, target.pdf(theta_p)*st.norm(0, sigma).pdf(theta)/[st.norm(0, sigma).pdf(theta_p)*target.pdf(theta)])
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
            samples.append(theta)
    return samples

### Random Walk MH

a = 10
b = 10
target = st.beta(a, b)
sigma = 0.3
naccept = 0
theta = 0.1
niters = 10000
samples = np.zeros(niters+1)
samples[0] = theta
for i in range(niters):
    theta_p = theta + st.norm(0, sigma).rvs()
    rho = min(1, target.pdf(theta_p)/target.pdf(theta))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
print naccept/niters


thetas = np.linspace(0, 1, 200)
plt.figure(figsize=(12, 9))
plt.hist(samples[nmcmc:], 40, histtype='step', normed=True, linewidth=1, label='Output distribution');
plt.plot(thetas, target.pdf(thetas), c='red', linestyle='--', label='True target density')
plt.xlim([0,1]);
plt.legend(loc='best');


### MALA

a = 10
b = 10
target = st.beta(a, b)
sigma = 0.3
naccept = 0
theta = 0.1
niters = 10000

delta = 0.01
gamma = 0.01


samples = np.zeros(niters+1)
samples[0] = theta
for i in range(niters):
    theta_1 = theta
    theta_2 = theta_1 + delta
    gradU = -(target.pdf(theta_2) - target.pdf(theta_1))/delta
    theta_p = theta +gamma*gradU + st.norm(0, sigma).rvs()
    rho = min(1, target.pdf(theta_p)*st.norm(0, sigma).pdf(theta)/[st.norm(0, sigma).pdf(theta_p)*target.pdf(theta)])
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
print naccept/niters

thetas = np.linspace(0, 1, 200)
plt.figure(figsize=(12, 9))
plt.hist(samples[nmcmc:], 40, histtype='step', normed=True, linewidth=1, label='Output distribution');
plt.plot(thetas, target.pdf(thetas), c='red', linestyle='--', label='True target density')
plt.xlim([0,1]);
plt.legend(loc='best');


#ULA

a = 10
b = 10
target = st.beta(a, b)
sigma = 0.3
naccept = 0
theta = 0.1
niters = 10000

delta = 0.01
gamma = 0.01


samples = np.zeros(niters+1)
samples[0] = theta
for i in range(niters):
    theta_1 = theta
    theta_2 = theta_1 + delta
    gradU = -(target.pdf(theta_2) - target.pdf(theta_1))/delta
    theta_p = theta +gamma*gradU + np.sqrt(2*gamma)*st.norm(0, sigma).rvs()
    naccept += 1
    theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
print naccept/niters

thetas = np.linspace(0, 1, 200)
plt.figure(figsize=(12, 9))
plt.hist(samples[nmcmc:], 40, histtype='step', normed=True, linewidth=1, label='Output distribution');
plt.plot(thetas, target.pdf(thetas), c='red', linestyle='--', label='True target density')
plt.xlim([0,1]);
plt.legend(loc='best');



#Compare MALA and MH
a = 10
b = 10
target = st.beta(a, b)
sigma = 0.3
sigma2 = 0.03
niters = 100

samples_mh = [mh_chain(niters, theta, target, sigma) for theta in np.arange(0.1, 4, 1)]
samples_mala = [mala_chain(niters, theta, target, sigma) for theta in np.arange(0.1, 4, 1)]

plt.figure(figsize=(20,10))
for samples in samples_mh:
    plt.plot(samples, '-')
for samples in samples_mala:
    plt.plot(samples, '-o')
plt.xlim([0, niters])
plt.ylim([0, 10]);




