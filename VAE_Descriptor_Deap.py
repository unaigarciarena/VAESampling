import tensorflow as tf

import numpy as np
import argparse
import time
import random
from tunable_gan_class_optimization import xavier_init, VAEES, VAEESDescriptor, VAEE, VAE
from Class_F_Functions import pareto_frontier, F_Functions, scale_columns, igd
import matplotlib.pyplot as plt

# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)

def plot(samples):


    nf1,nf2 = MOP_f.Evaluate_MOP_Function(samples)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f(x_1)$')
    ax.set_ylabel('$f(x_2)$')
    plt.plot(pf1, pf2, 'b.')
    plt.plot(nf1, nf2, 'r.')

    plt.show()


All_Evals = 0
descriptors = []
descriptors_e = []
descriptors_e_s = []


def create_descriptor():

    encoder_n_hidden = np.random.randint(n_layers)+1                             # Number of hidden layers
    decoder_n_hidden = np.random.randint(n_layers)+1                             # Number of hidden layers
    approx_n_hidden = np.random.randint(n_layers)+1

    encoder_dim_list = [np.random.randint(max_layer_size)+1 for _ in range(encoder_n_hidden)]
    decoder_dim_list = [np.random.randint(max_layer_size)+1 for _ in range(decoder_n_hidden)]
    approx_dim_list = [np.random.randint(max_layer_size)+1 for _ in range(approx_n_hidden)]

    f_measure = divergence_measures[np.random.randint(len(divergence_measures))]

    i_g_function = init_functions[np.random.randint(len(init_functions))]   # List or random init functions for encoder
    i_d_function = init_functions[np.random.randint(len(init_functions))]   # List or random init functions for decoder
    i_a_function = init_functions[np.random.randint(len(init_functions))]   # List or random init functions for decoder

    encoder_init_functions = []
    decoder_init_functions = []
    approx_init_functions = []

    encoder_act_functions = []
    decoder_act_functions = []
    approx_act_functions = []

    for i in range(encoder_n_hidden+1):
        encoder_init_functions.append(i_g_function)
        if i == encoder_n_hidden:
            encoder_act_functions.append(None)        # Activation functions for all layers in encoder
        else:
            encoder_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    for i in range(decoder_n_hidden+1):

        decoder_init_functions.append(i_d_function)
        if i == decoder_n_hidden:
            decoder_act_functions.append(None)        # Activation functions for all layers in decoder
        else:
            decoder_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    for i in range(approx_n_hidden+1):

        approx_init_functions.append(i_a_function)
        if i == approx_n_hidden:
            approx_act_functions.append(None)        # Activation functions for all layers in approximator
        else:
            approx_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    latent_distribution_function = lat_functions[0]   # (Only Normal) lat_functions[np.random.randint(len(lat_functions))]

    my_vae_descriptor = VAEESDescriptor(X_dim, z_dim, objectives, latent_distribution_function, f_measure)

    my_vae_descriptor.vae_decoder_initialization(encoder_n_hidden, encoder_dim_list, encoder_init_functions,  encoder_act_functions)

    my_vae_descriptor.vae_encoder_initialization(decoder_n_hidden, decoder_dim_list, decoder_init_functions,  decoder_act_functions)

    my_vae_descriptor.vae_approximator_initialization(approx_n_hidden, approx_dim_list, approx_init_functions,  approx_act_functions)

    return my_vae_descriptor


def main():
    global All_Evals
    global best_val
    global descriptors
    global descriptors_e
    global descriptors_e_s
    reset_graph()

    while True:

        my_vae_e_s_descriptor = create_descriptor()
        """
        time_e_s = time.time()
        my_vae_e_s = VAEES(my_vae_e_s_descriptor)
        my_vae_e_s.training_definition(lr)
        final_samples_e_s = my_vae_e_s.running("fixed", ps_fs, mb_size, number_epochs, number_samples, print_cycle)
        time_e_s = time.time() - time_e_s
        nf1_e_s, nf2_e_s = MOP_f.Evaluate_MOP_Function(final_samples_e_s)
        tf1_e_s, tf2_e_s = pareto_frontier(nf1_e_s, nf2_e_s)
        igd_val_e_s = igd(np.vstack((tf1_e_s, tf2_e_s)).transpose(), np.vstack((pf1, pf2)).transpose())

        if np.isnan(final_samples_e_s).any():
            continue
        """

        my_vae_e_s_descriptor.Dec_network.input_dim = z_dim

        time_ = time.time()
        my_vae = VAE(my_vae_e_s_descriptor)
        my_vae.training_definition(lr)
        final_samples = my_vae.running("fixed", ps_all_x, mb_size, number_epochs,  number_samples, print_cycle)
        print(final_samples)
        plot(final_samples)
        time_ = time.time() - time_
        nf1, nf2 = MOP_f.Evaluate_MOP_Function(final_samples)
        tf1, tf2 = pareto_frontier(nf1, nf2)
        igd_val = igd(np.vstack((tf1, tf2)).transpose(), np.vstack((pf1, pf2)).transpose())

        if np.isnan(final_samples).any():
            continue
        else:
            break

        time_e = time.time()
        my_vae_e = VAEE(my_vae_e_s_descriptor)
        my_vae_e.training_definition(lr)
        final_samples_e = my_vae_e.running("fixed", ps_fs, mb_size, number_epochs, number_samples, print_cycle)
        time_e = time.time() - time_e
        nf1_e, nf2_e = MOP_f.Evaluate_MOP_Function(final_samples_e)
        tf1_e, tf2_e = pareto_frontier(nf1_e, nf2_e)
        igd_val_e = igd(np.vstack((tf1_e, tf2_e)).transpose(), np.vstack((pf1, pf2)).transpose())

        if np.isnan(final_samples).any():
            continue
        else:
            break

    descriptors += [[igd_val, time_]+my_vae_e_s_descriptor.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions)]
    #descriptors_e += [[igd_val_e, time_e]+my_vae_e_s_descriptor.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions)]
    #descriptors_e_s += [[igd_val_e_s, time_e_s] + my_vae_e_s_descriptor.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions)]

#####################################################################################################

                 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(3000), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    myseed = args.integers[0]               # Seed: Used to set different outcomes of the stochastic program
    number_samples = args.integers[1]       # Number samples to generate by encoder  (number_samples=1000)
    X_dim = args.integers[2]                  # Number of Variables of the MO function (n=10)
    Function = "F"+str(args.integers[3])  # MO function to optimize, all in F1...F9, except F6
    z_dim = args.integers[4]               # Dimension of the latent variable  (z_dim=30)
    n_layers = args.integers[5]              # Maximum number of layers for generator and discriminator (n_layers = 10)
    max_layer_size = args.integers[6]       # Maximum size of the layers  (max_layer_size = 50)

    random.seed(myseed)
    tf.set_random_seed(myseed)
    np.random.seed(myseed)

    k = 1000                          # Number of samples of the Pareto set
    objectives = 2
    MOP_f = F_Functions(X_dim, Function)
    ps_all_x = MOP_f.Generate_PS_samples(k)
    pf1, pf2 = MOP_f.Evaluate_MOP_Function(ps_all_x)

    ps_all_x = scale_columns(ps_all_x)
    print(ps_all_x)
    ps_fs = np.hstack((ps_all_x, np.reshape(pf1, (-1, 1)), np.reshape(pf2, (-1, 1))))

    act_functions = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
    init_functions = [xavier_init, tf.random_uniform, tf.random_normal]
    lat_functions = [np.random.normal]
    divergence_measures = ["VAE_MSE_Div"]
    mutation_types = ["add_layer", "del_layer", "weigt_init", "activation", "dimension", "divergence"]
    crossover_types = ["encoder", "approximator", "decoder", "encoder+approximator", "decoder+approximator", "encoder+decoder"]
    mutation_types_restricted = ["weigt_init", "activation", "dimension", "divergence"]

    mb_size = 50                                              # Minibatch size
    number_epochs = 1001                                      # Number epochs for training
    print_cycle = 250                                         # Frequency information is printed
    lr = 1e-2                                                  # Learning rate for Adam optimizer

    descriptors = []
    descriptors_e = []
    descriptors_e_s = []
    number_epochs = 501                                      # Number epochs for training
    for _ in range(50):
        main()

    np.savetxt(fname="results/VAE1500" + str(myseed) + Function + ".csv", X=descriptors)
    np.savetxt(fname="results/VAE_E1500" + str(myseed) + Function + ".csv", X=descriptors_e)
    np.savetxt(fname="results/VAE_E_S1500" + str(myseed) + Function + ".csv", X=descriptors_e_s)

