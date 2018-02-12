
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
import matplotlib.pyplot as plt
from Class_F_Functions import pareto_frontier, F_Functions, scale_columns, igd
from tunable_gan_class_optimization import Network, NetworkDescriptor, VAEESDescriptor

#np.random.seed(2)
#tf.set_random_seed(2)

def batch(x, size, i):
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
        return index, np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return index, x[i:index, :]


n_samples = 1000
X_dim = 784
Function = "F8"
MOP_f = F_Functions(X_dim, Function)
ps_all_x = MOP_f.Generate_PS_samples(n_samples)
pf1, pf2 = MOP_f.Evaluate_MOP_Function(ps_all_x)

ps_all_x = scale_columns(ps_all_x)

# In[3]:

def plot(samples):


    nf1,nf2 = MOP_f.Evaluate_MOP_Function(samples)

    igd_val = igd(np.vstack((nf1,nf2)).transpose(), np.vstack((pf1,pf2)).transpose())
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f(x_1)$')
    ax.set_ylabel('$f(x_2)$')
    plt.plot(pf1, pf2, 'b.')
    plt.plot(nf1, nf2, 'r.')
    print(igd_val)
    plt.show()

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


# Based on this, we define now a class "VariationalAutoencoder" with a [sklearn](http://scikit-learn.org)-like interface that can be trained incrementally with mini-batches using partial_fit. The trained model can be used to reconstruct unseen input, to generate new samples, and to map inputs to the latent space.

# In[4]:

#class VAEDescriptor:


class VAE(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture.X_dim])

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
        self.layer_21 = self.gen_network.network_evaluation(self.z)
        return self.layer_21

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                                       + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, x):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        _, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: x})

        return cost

    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: x})

    def generate(self, samples):
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


# In general, implementing a VAE in tensorflow is relatively straightforward (in particular since we don not need to code the gradient computation). A bit confusing is potentially that all the logic happens at initialization of the class (where the graph is generated), while the actual sklearn interface methods are very simple one-liners.
#
# We can now define a simple fuction which trains the VAE using mini-batches:

# In[5]:

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):

    vae_desc = VAEESDescriptor(784, 500, 2, [np.random.normal], ["VAE_MSE_Div"])
    vae_desc.vae_decoder_initialization(2, [500, 500], [xavier_init, xavier_init, xavier_init], [tf.nn.softplus, tf.nn.softplus, tf.nn.sigmoid], None)
    vae_desc.vae_encoder_initialization(1, [500], [xavier_init, xavier_init], [tf.nn.softplus, tf.nn.softplus], None)

    vae = VAE(vae_desc, learning_rate=learning_rate, batch_size=batch_size)
    # Training cycle
    batch_i = 0
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for _ in range(total_batch):
            batch_i, batch_xs = batch(ps_all_x, batch_size, batch_i)
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


# ## Illustrating reconstruction quality

# We can now train a VAE on MNIST by just specifying the network topology. We start with training a VAE with a 20-dimensional latent space.

# In[6]:

network_architecture = dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
                            n_hidden_recog_2=500,  # 2nd layer encoder neurons
                            n_hidden_gener_1=500,  # 1st layer decoder neurons
                            n_hidden_gener_2=500,  # 2nd layer decoder neurons
                            n_input=784,  # MNIST data input (img shape: 28*28)
                            n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=300)


# Based on this we can sample some test inputs and visualize how well the VAE can reconstruct those. In general the VAE does really well.

# In[7]:

_, x_sample = batch(ps_all_x, 100, 0)
x_reconstruct = vae.reconstruct(x_sample)
if False:
    plt.figure(figsize=(8, 12))
    for i in range(5):

        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.show()

aux = vae.generate(1000)

plot(aux)

print(MOP_f.Evaluate_MOP_Function(aux[:2]))

print(aux)
