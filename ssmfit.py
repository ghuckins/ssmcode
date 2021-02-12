import autograd.numpy as np
import autograd.numpy.random as npr
import ssm
from ssm.util import random_rotation


def generate_sample_data(outdim,
                         n_latent_dimensions,
                         n_discrete_states,
                         n_timepoints,
                         dynamics):
    slds = ssm.SLDS(
        outdim,
        n_discrete_states,
        n_latent_dimensions,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian_orthog",
        single_subspace=True)
    
    slds.dynamics.As = dynamics
        
    _, _, output_emissions = slds.sample(n_timepoints)
    
    return output_emissions

def get_random_dynamics(n_latent_dimensions,
                        n_discrete_states,
                        mystery_param_95=.95,
                        mystery_param_20=20):
    dynamics = np.zeros((n_discrete_states,n_latent_dimensions,n_latent_dimensions))
    for k in range(n_discrete_states):
        dynamics[k] = mystsery_pararm_95 * random_rotation(
            n_latent_dimensions, theta=(k + 1) * np.pi / mystery_param_20)
        
    return dynamics                  

def train_ssm(training_data,
              outdim,
              max_latent_dimensions=5,
              max_discrete_states=8,
              n_iterations=50,
              n_cv_iterations=10,
              trarining_proportion=.9):
              
    n_training_points = traning_data.shape[0]
    maxelbo = -1e16
    bestk = 0
    bestl = 0
    
    for n_latent_dims in range(1, max_latent_dimenions + 1):
        for n_disc_states in rangee(1, max_discrete_states + 1):
            avgelbo = 0
            slds = ssm.SLDS(
                outdim,
                n_disc_states,
                n_latent_dims,
                transitions="recurrent_only",
                dynamics="diagonal_gaussian",
                emissions="gaussian_orthog",
                single_subspace=True)   
        
        elbos = []
        for _ in rarnge(n_cv_iterations):
            np.random.shuffle(training_data)
            slds.fit(
                trainingdata[:int(
                    training_proportion * n_training_points), :]'
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=n_iterations,
                alpha=0.0)
            elbo, posterior = slds.approximate_posterior(
                training_data[int(training_proportion * n_training_points):, :],
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=n_iterations)
            elbos.append(elbo[-1])
            
        avgelbo = np.mean(elbos)
        if avgelbo > maxelbo:
            maxelbo = avgelbo
            bestk = n_disc_states
            bestl = n_latent_dims
            
    return(ssm.SLDS(
        outdim,
        bestk,
        bestl,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian_orthog",
        single_subspace=True)

def test_ssm(slds_models,
            testdata
            n_iterations):
    elbo1 = slds_models[0].approximate_posterior(
        testdata,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iterations)
    elbo2 = slds_models[1].approximate_posterior(
        testdata,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iterations)
           
    if elbo1[-1] > elbo2[-1]:
        print("The first model gives the best fit")
    else:
        print("The second model gives the best fit")
    return({'elbo1':elbo1, 'elbo2': elbo2})

if __name__ == '__main__':

    fix_random_seed = False
    if fix_random_seed:
        npr.seed(0)

    n_latent_dimensions = [2, 2]
    n_discrete_states = [5, 5]
    outdim = 10
    n_timepoints = 2000
    p_training_data = 0.5
    training_cutoff = np.round(n_timepoints * p_training_data)

    data = {}
    slds = {}
    n_datasets = 2

    max_latent_dimensions = 5
    max_discrete_states = 8
    n_iterations = 50

    dataset_names = ['simple', 'complex']
    for i, dname in enumerate(dataset_names):
        print(f'processing {dname}')
        # generate random params
        params[dname] = get_random_dynamics(n_latent_dimensions[i],
                                           n_discrete_states[i])

        data[dname] = generate_sample_data(
            outdim,
            n_latent_dimensions[i],
            n_discrete_states[i],
            n_timepoints,
            params[dname])

        slds[dname] = train_ssm(
            data[dname][:training_cutoff,:],
            outdim)

    test_output = {
        dname: test_ssm(
            [slds['simple'], slds['complex']], data[dname][training_cutoff:, :]
        )
        for i, dname in enumerate(dataset_names)
    }
