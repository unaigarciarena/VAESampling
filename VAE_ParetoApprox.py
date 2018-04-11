
# coding: utf-8

# # Variational Autoencoder in TensorFlow

# The main motivation for this post was that I wanted to get more experience with both [Variational Autoencoders](http://arxiv.org/abs/1312.6114) (VAEs) and with [Tensorflow](http://www.tensorflow.org/). Thus, implementing the former in the latter sounded like a good idea for learning about both at the same time. This post summarizes the result.
#
# Note: The post was updated on December 7th 2015:
#   * a bug in the computation of the latent_loss was fixed (removed an erroneous factor 2). Thanks Colin Fang for pointing this out.
#   * Using a Bernoulli distribution rather than a Gaussian distribution in the generator network
#
# Note: The post was updated on January 3rd 2017:
#   * changes required for supporting TensorFlow v0.12 and Python 3 support
#
# Let us first do the necessary imports, load the data (MNIST), and define some helper functions.

# In[1]:

import numpy as np
import tensorflow as tf

from Class_F_Functions import F_Functions, scale_columns, igd
from tunable_gan_class_optimization import VAEESDescriptor, VAE, VAEE, VAEEE
import argparse
import time


def batch(x, y, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param y: Fitness scores of the population, intended to be fed to the net in the output
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array
        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return index, np.concatenate((x[i:, :], x[:index, :])), np.concatenate((y[i:], y[:index]))
    else:  # Easy case
        index = i+size
        return index, x[i:index, :], y[i:index]


def plot(samples):

    nf1, nf2 = MOP_f.Evaluate_MOP_Function(samples)

    igd_value = igd(np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose())
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f(x_1)$')
    ax.set_ylabel('$f(x_2)$')
    plt.plot(pf1, pf2, 'b.')
    plt.plot(nf1, nf2, 'r.')
    plt.show()
    """
    return igd_value


def xavier_init(fan_in=None, fan_out=None, shape=None, constant=1):
    """ Xavier initialization of network weights"""
    if fan_in is None:
        fan_in = shape[0]
        fan_out = shape[1]
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


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
            encoder_act_functions.append(None)
        else:
            encoder_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    for i in range(decoder_n_hidden+1):

        decoder_init_functions.append(i_d_function)
        if i == decoder_n_hidden:
            decoder_act_functions.append(tf.sigmoid)
        else:
            decoder_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    for i in range(approx_n_hidden+1):

        approx_init_functions.append(i_a_function)
        if i == approx_n_hidden:
            approx_act_functions.append(None)
        else:
            approx_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    latent_distribution_function = lat_functions[0]   # (Only Normal) lat_functions[np.random.randint(len(lat_functions))]

    my_vae_descriptor = VAEESDescriptor(X_dim, z_dim, objectives, latent_distribution_function, f_measure)

    my_vae_descriptor.vae_encoder_initialization(encoder_n_hidden, encoder_dim_list, encoder_init_functions,  encoder_act_functions)

    my_vae_descriptor.vae_decoder_initialization(decoder_n_hidden, decoder_dim_list, decoder_init_functions,  decoder_act_functions)

    my_vae_descriptor.vae_approximator_initialization(approx_n_hidden, approx_dim_list, approx_init_functions,  approx_act_functions)

    if type == "VAE":
            my_vae_descriptor.Dec_network.input_dim = my_vae_descriptor.z_dim
    if type == "VAEE":
            my_vae_descriptor.Dec_network.input_dim = my_vae_descriptor.z_dim
            my_vae_descriptor.App_network.input_dim = my_vae_descriptor.z_dim

    return my_vae_descriptor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(3000), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    myseed = args.integers[0] - 1         # Seed: Used to set different outcomes of the stochastic program
    n_samples = args.integers[1]          # Number samples to generate by encoder  (number_samples=1000)
    X_dim = args.integers[2]              # Number of Variables of the MO function (n=10)
    Function = "F"+str(args.integers[3])  # MO function to optimize, all in F1...F9, except F6
    z_dim = args.integers[4]              # Dimension of the latent variable  (z_dim=30)
    n_layers = args.integers[5]           # Maximum number of layers for generator and discriminator (n_layers = 10)
    max_layer_size = args.integers[6]     # Maximum size of the layers  (max_layer_size = 50)
    training_epochs = args.integers[7]    # Training iterations for each VAE instance (training_epochs = 300)
    divergence_measures = ["VAE_MSE_Div", "VAE_Log_Prob"]
    init_functions = [tf.random_uniform, tf.random_normal, xavier_init]
    act_functions = [tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh, None]
    lat_functions = [np.random.normal]
    objectives = 2
    np.seterr(all="raise")

    MOP_f = F_Functions(X_dim, Function)
    ps_all_x = MOP_f.Generate_PS_samples(n_samples)
    pf1, pf2 = MOP_f.Evaluate_MOP_Function(ps_all_x)
    objs = np.array([pf1, pf2])
    objs = np.transpose(objs)
    ps_all_x = scale_columns(ps_all_x)

    vae_cod = []
    vae_e_cod = []
    vae_ee_cod = []
    while myseed < 550:
        print("Seed: " + str(myseed))
        np.random.seed(myseed)
        tf.set_random_seed(myseed)
        tf.reset_default_graph()

        vae_desc = create_descriptor()

        start = time.time()
        vae = VAEEE(vae_desc, objs.shape[1])

        vae.train(epochs=training_epochs, objs=objs, data=ps_all_x)

        aux = vae.generate(1000, objs)[0]
        print(aux)
        igd_val = plot(aux)
        vae_ee_cod += [[myseed, time.time()-start] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]

        print("VAE_EE IGD: " + str(igd_val))

        vae_desc.Dec_network.input_dim = z_dim
        vae_desc.App_network.input_dim = z_dim

        start = time.time()
        vae = VAEE(vae_desc, objs.shape[1])
        vae.train(epochs=training_epochs, objs=objs, data=ps_all_x)

        aux = vae.generate(1000, objs)[0]
        igd_val = plot(aux)
        vae_e_cod += [[myseed, time.time()-start] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]

        print("VAE_E IGD: " + str(igd_val))

        start = time.time()

        vae = VAE(vae_desc, objs.shape[1])
        vae.train(epochs=training_epochs, objs=None, data=ps_all_x)
        _, x_sample, y_sample = batch(ps_all_x, objs, 50, 0)

        aux = vae.generate(1000, objs)
        igd_val = plot(aux)
        vae_cod += [[myseed, time.time()-start] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]
        print("VAE IGD: " + str(igd_val))

        myseed += 50

    np.savetxt("results/VAE_E_F" + Function + "seed" + str(myseed) + ".csv", vae_e_cod)
    np.savetxt("results/VAE_EE_F" + Function + "seed" + str(myseed) + ".csv", vae_ee_cod)
    np.savetxt("results/VAE_F" + Function + "seed" + str(myseed) + ".csv", vae_cod)
