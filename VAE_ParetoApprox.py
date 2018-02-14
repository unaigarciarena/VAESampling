
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
from tunable_gan_class_optimization import Network, VAEESDescriptor
import argparse



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


class VAE(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Div_Function = self.Divergence_Functions[self.network_architecture.fmeasure]
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture.X_dim])
        self.reconstr_loss = None
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = self._recognition_network()

        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._generator_network()

    def _recognition_network(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        recog_descriptor = self.network_architecture.Enc_network

        w_mean = tf.Variable(xavier_init(recog_descriptor.output_dim, self.network_architecture.z_dim))
        w_log_sigma = tf.Variable(xavier_init(recog_descriptor.output_dim, self.network_architecture.z_dim))
        b_mean = tf.Variable(tf.zeros([self.network_architecture.z_dim], dtype=tf.float32)),
        b_log_sigma = tf.Variable(tf.zeros([self.network_architecture.z_dim], dtype=tf.float32))

        self.recog_network = Network(recog_descriptor)
        self.recog_network.network_initialization()
        layer_2 = self.recog_network.network_evaluation(self.x)
        z_mean = tf.add(tf.matmul(layer_2, w_mean), b_mean)
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, w_log_sigma), b_log_sigma)
        return z_mean, z_log_sigma_sq

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        gen_descriptor = self.network_architecture.Dec_network
        self.gen_network = Network(gen_descriptor)
        self.gen_network.network_initialization()
        x_reconstr_mean = self.gen_network.network_evaluation(self.z)
        return x_reconstr_mean

    def _create_loss_optimizer(self):

        self.Div_Function(self)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(self.reconstr_loss + latent_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def vae_mse_div(self):
        # E[log P(X|z)]
        self.reconstr_loss = 1000*tf.losses.mean_squared_error(predictions=self.x_reconstr_mean, labels=self.x)

    def vae_log_prob(self):
        self.reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

    Divergence_Functions = {"VAE_Log_Prob": vae_log_prob, "VAE_MSE_Div": vae_mse_div}

    def partial_fit(self, x, _):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        _, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: x})

        return cost

    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: x})

    def generate(self, samples, _):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """

        z_mu = np.random.normal(size=(samples, self.network_architecture.z_dim))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: np.reshape(z_mu, (samples, -1))})

    def reconstruct(self, x):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: x})


def train(batch_size=100, training_epochs=10, display_step=299):

    # Training cycle
    batch_i = 0
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for _ in range(total_batch):
            batch_i, batch_xs, batch_ys = batch(ps_all_x, objs, batch_size, batch_i)
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs, batch_ys)

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

class VAE_E(VAE):

    def __init__(self, network_architecture, objectives):

        self.objectives = objectives
        self.approximation = None
        self.obj = tf.placeholder(tf.float32, shape=[None, self.objectives.shape[1]])

        super().__init__(network_architecture)


    def _approximation_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        app_descriptor = self.network_architecture.App_network
        self.app_network = Network(app_descriptor)
        self.app_network.network_initialization()
        approximation = self.app_network.network_evaluation(self.z)

        return approximation

    def _create_network(self):

        super()._create_network()
        self.approximation = self._approximation_network()


    def _create_loss_optimizer(self):

        self.Div_Function(self)

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.app_cost = 10*tf.losses.mean_squared_error(predictions=self.approximation, labels=self.obj)

        self.cost = tf.reduce_mean(self.reconstr_loss + latent_loss + self.app_cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def predict_from_z(self, samples):

        z_mu = np.random.normal(size=(samples, self.network_architecture.z_dim))
        return self.sess.run(self.approximation, feed_dict={self.z: np.reshape(z_mu, (samples, -1))})

    def predict_from_x(self, x):
        return self.sess.run(self.approximation, feed_dict={self.x: x})

    def partial_fit(self, x, obj):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        _, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: x, self.obj: obj})

        return cost

class VAE_E_E(VAE_E):

    def __init__(self, network_architecture, objectives):
        super().__init__(network_architecture, objectives)

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        gen_descriptor = self.network_architecture.Dec_network
        self.gen_network = Network(gen_descriptor)
        self.gen_network.network_initialization()
        x_reconstr_mean = self.gen_network.network_evaluation(tf.concat((self.z, self.obj), axis=1))
        return x_reconstr_mean

    def _approximation_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        app_descriptor = self.network_architecture.App_network
        self.app_network = Network(app_descriptor)
        self.app_network.network_initialization()

        approximation = self.app_network.network_evaluation(tf.concat((self.z, self.obj), axis=1))
        return approximation

    def generate(self, samples, objs):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """

        z_mu = np.random.normal(size=(samples, self.network_architecture.z_dim))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution

        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: np.reshape(z_mu, (samples, -1)), self.obj: objs[np.random.randint(objs.shape[0], size=objs.shape[0]), :]})

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

    return my_vae_descriptor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(3000), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    myseed = args.integers[0]               # Seed: Used to set different outcomes of the stochastic program
    n_samples = args.integers[1]       # Number samples to generate by encoder  (number_samples=1000)
    X_dim = args.integers[2]                  # Number of Variables of the MO function (n=10)
    Function = "F"+str(args.integers[3])  # MO function to optimize, all in F1...F9, except F6
    z_dim = args.integers[4]               # Dimension of the latent variable  (z_dim=30)
    n_layers = args.integers[5]              # Maximum number of layers for generator and discriminator (n_layers = 10)
    max_layer_size = args.integers[6]       # Maximum size of the layers  (max_layer_size = 50)

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

    myseed = 0
    vae_cod = []
    vae_e_cod = []
    while myseed < 51:
        print("Seed: " + str(myseed))
        np.random.seed(myseed)
        tf.set_random_seed(myseed)
        tf.reset_default_graph()

        vae_desc = create_descriptor()
        #print(vae_desc.App_network.input_dim)
        vae = VAE_E_E(vae_desc, objs)

        vae = train(training_epochs=300)

        _, x_sample, y_sample = batch(ps_all_x, objs, 50, 0)
        aux = vae.generate(1000, objs)
        igd_val = plot(aux)
        vae_e_cod += [[myseed] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]

        print("VAE_E_E IGD: " + str(igd_val))

        vae_desc.Dec_network.input_dim = z_dim
        vae_desc.App_network.input_dim = z_dim


        vae = VAE_E(vae_desc, objs)
        vae = train(training_epochs=300)

        _, x_sample, y_sample = batch(ps_all_x, objs, 50, 0)
        aux = vae.generate(1000, None)
        igd_val = plot(aux)
        vae_e_cod += [[myseed] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]

        print("VAE_E IGD: " + str(igd_val))




        #print(np.concatenate((vae.predict_from_x(x_sample), y_sample), axis=1))
        vae = VAE(vae_desc, learning_rate=0.001)
        vae = train(training_epochs=300)
        _, x_sample, y_sample = batch(ps_all_x, objs, 50, 0)

        aux = vae.generate(1000, None)
        igd_val = plot(aux)
        vae_cod += [[myseed] + vae.network_architecture.codify_components(n_layers, init_functions, act_functions, divergence_measures, lat_functions) + [igd_val]]
        print("VAE IGD: " + str(igd_val))
        myseed += 50


    np.savetxt("VAE_E" + str(myseed) + ".csv", vae_e_cod)
    np.savetxt("VAE" + str(myseed) + ".csv", vae_cod)
