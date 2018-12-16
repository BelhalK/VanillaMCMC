import numpy,math
import matplotlib.pyplot as plt
import random as random

random.seed(2)

# Draw random samples
def sample(N, alpha, M_min, M_max):
    log_M_Min = math.log(M_min)
    log_M_Max = math.log(M_max)
    maxlik = math.pow(M_min, 1.0 - alpha)

    Masses = []
    while (len(Masses) < N):
        logM = random.uniform(log_M_Min,log_M_Max)
        M  = math.exp(logM)
        likelihood = math.pow(M, 1.0 - alpha)
        u = random.uniform(0.0,maxlik)
        if (u < likelihood):
            Masses.append(M)
    return Masses

# def logpdf():
#     return 

# def logpdf_normal():
#     return 

# def kull_div():
#     return 

# def compare_kl_chi():
#     return 





# Draw samples.
Masses = sample(1000000, 2.35, 1.0, 100.0)
# Convert to logM.
LogMasses = numpy.log(numpy.array(Masses))



# Plot distribution.
plt.figure(1)
plt.hist(LogMasses, 30, histtype='step', lw=3, log=True,
         range=(0.0,math.log(100.0)))
# plot the samples and the logpdf
X = []
Y = []
for n in range(101):
    logM = math.log(100.0)*float(n)/100.0
    x    = math.exp(logM)
    y    = 2.0e5*math.pow(x, 1.0-2.35)  # normalisation
    X.append(logM)
    Y.append(y)
plt.plot(X, Y, '-', lw=3, color='black')
plt.xlim(0.0,math.log(100.0))
plt.xlabel(r'$\log M$', fontsize=24)
plt.ylabel('PDF', fontsize=24)
plt.show()



# LogLikelihood
def LogLikelihood(params, D, N, M_min, M_max):
    alpha = params[0]  # extract alpha
    # Compute normalisation constant.
    c = (1.0 - alpha)/(math.pow(M_max, 1.0-alpha)
                        - math.pow(M_min, 1.0-alpha))
    # return log likelihood.
    return N*math.log(c) - alpha*D

# Generate toy data.
N      = 1000000  # Draw 1 Million stellar masses.
alpha  = 2.35
M_min  = 1.0
M_max  = 100.0
Masses = sample(N, alpha, M_min, M_max)
LogM   = numpy.log(numpy.array(Masses))
D      = numpy.mean(LogM)*N


# guess alpha
guess = [3.0]
A = [guess]
stepsizes = [0.005]  
accepted  = 0.0

# MH algo
for n in range(10000):
    old_alpha  = A[len(A)-1] 
    old_loglik = LogLikelihood(old_alpha, D, N, M_min,
                    M_max)
    # New proposal distribution
    new_alpha = numpy.zeros([len(old_alpha)])
    for i in range(len(old_alpha)):
        new_alpha[i] = random.gauss(old_alpha[i], stepsizes[i])
    new_loglik = LogLikelihood(new_alpha, D, N, M_min,
                    M_max)
    # New candidate sample
    if (new_loglik > old_loglik):
        A.append(new_alpha)
        accepted = accepted + 1.0  #Acceptance rate
    else:
        u = random.uniform(0.0,1.0)
        if (u < math.exp(new_loglik - old_loglik)):
            A.append(new_alpha)
            accepted = accepted + 1.0  
        else:
            A.append(old_alpha)

print "Acceptance rate = "+str(accepted/10000.0)

#Get rid of half of the chain (consider when the chain has converged)
Clean = []
for n in range(5000,10000):
    if (n % 10 == 0):
        Clean.append(A[n][0])


# alpha estimate
print "Mean:  "+str(numpy.mean(Clean))
print "Sigma: "+str(numpy.std(Clean))

plt.figure(1)
plt.hist(Clean, 20, histtype='step', lw=3)
plt.xticks([2.346,2.348,2.35,2.352,2.354],
           [2.346,2.348,2.35,2.352,2.354])
plt.xlim(2.345,2.355)
plt.xlabel(r'$\alpha$', fontsize=24)
plt.ylabel(r'$\cal L($Data$;\alpha)$', fontsize=24)
plt.savefig('example-MCMC-results.png')
plt.show()


