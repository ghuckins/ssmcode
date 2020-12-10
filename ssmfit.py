import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import ssm
from ssm.util import random_rotation

def generate_sample_data(outdim, latentdim1, k1, latentdim2, k2,t):
    slds1 = ssm.SLDS(outdim, k1, latentdim1, transitions="recurrent_only", dynamics="diagonal_gaussian", emissions="gaussian_orthog", single_subspace=True)
    for k in range(k1):
        slds1.dynamics.As[k] = .95 * random_rotation(latentdim1, theta=(k + 1) * np.pi / 20)
    slds2 = ssm.SLDS(outdim, k2, latentdim2, transitions="recurrent_only", dynamics="diagonal_gaussian", emissions="gaussian_orthog", single_subspace=True)
    for k in range(k2):
        slds2.dynamics.As[k] = .95 * random_rotation(latentdim2, theta=(k + 1) * np.pi / 20)
    z,x,y1 = slds1.sample(t)
    z,x,y2 = slds2.sample(t)
    return y1,y2

def train_ssm(train,outdim,latentdims,ks,numits):
    size = np.shape(train)[0]
    maxelbo = -100000000000000000000
    bestk = 0
    bestl = 0
    for l in latentdims:
        for k in ks:
            avgelbo = 0
            slds = ssm.SLDS(outdim,k,l,transitions="recurrent_only", dynamics="diagonal_gaussian", emissions="gaussian_orthog", single_subspace=True)
            for i in range(10):
                np.random.shuffle(train)
                slds.fit(train[:int(.9*size),:],method="laplace_em",variational_posterior="structured_meanfield",num_iters=numits, alpha=0.0)
                elbo,posterior = slds.approximate_posterior(train[int(.9*size):,:],method="laplace_em",variational_posterior="structured_meanfield",num_iters=numits)
                avgelbo = (avgelbo*i + elbo[numits])/(i+1)
            if avgelbo > maxelbo:
                maxelbo = avgelbo
                bestk = k
                bestl = l
    slds = ssm.SLDS(outdim, bestk, bestl, transitions="recurrent_only", dynamics="diagonal_gaussian", emissions="gaussian_orthog", single_subspace=True)
    return slds

def test_ssm(slds1,slds2,test,numits):
    elbo1 = slds1.approximate_posterior(test,method="laplace_em",variational_posterior="structured_meanfield",num_iters=numits)
    elbo2 = slds2.approximate_posterior(test,method="laplace_em",variational_posterior="structured_meanfield",num_iters=numits)
    if elbo1[50] > elbo2[50]:
        print("The first model gives the best fit")
    else:
        print("The second model gives the best fit")


data1,data2 = generate_sample_data(10,2,5,2,5,1000)
slds1 = train_ssm(data1[:900,:],10,range(1,5),range(1,8),50)
slds2 = train_ssm(data2[:900,:],10,range(1,5),range(1,8),50)
test_ssm(slds1,slds2,data1[900:,:])
