import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import copy


###################################################################################################
# ######## Auxiliary Functions
###################################################################################################

def plot(samples, theshape):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:25, :]):  # enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # plt.imshow(sample.reshape(theshape), cmap='Greys_r')
        plt.imshow(sample.reshape(theshape))

    return fig


def next_random_batch(num, data):

    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = np.array(data)[idx]
    return data_shuffle


def next_batch(num, data, start):

    """
    Return a total of `num` samples and labels.
    """
    idx = np.arange(start, np.min([start+num, len(data)]))

    return np.array(data)[idx]

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
        if y is not None:
            return index, np.concatenate((x[i:, :], x[:index, :])), np.concatenate((y[i:], y[:index]))
        else:
            return index, np.concatenate((x[i:, :], x[:index, :])), None
    else:  # Easy case
        index = i+size
        if y is not None:
            return index, x[i:index, :], y[i:index]
        else:
            return index, x[i:index, :], None



def xavier_init(fan_in=None, fan_out=None, shape=None, constant=1):
    """ Xavier initialization of network weights"""
    if fan_in is None:
        fan_in = shape[0]
        fan_out = shape[1]
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((int(fan_in), int(fan_out)),
                             minval=low, maxval=high)


# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)

###############################################################################################################################
# ########################################################## Network Descriptor #######################################################################################################################################################################################


class NetworkDescriptor:
    def __init__(self, number_hidden_layers, input_dim, output_dim,  list_dims, list_init_functions, list_act_functions, number_loop_train):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim 
        self.output_dim = output_dim  
        self.List_dims = list_dims
        self.List_init_functions = list_init_functions
        self.List_act_functions = list_act_functions
        self.number_loop_train = number_loop_train

    def copy_from_othernetwork(self, other_network):       
        self.number_hidden_layers = other_network.number_hidden_layers
        self.input_dim = other_network.input_dim
        self.output_dim = other_network.output_dim  
        self.List_dims = copy.deepcopy(other_network.List_dims)
        self.List_init_functions = copy.deepcopy(other_network.List_init_functions)
        self.List_act_functions = copy.deepcopy(other_network.List_act_functions)
        self.number_loop_train = other_network.number_loop_train

    def network_add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function):

        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.List_dims.insert(layer_pos, lay_dims)
        self.List_init_functions.insert(layer_pos, init_w_function)
        self.List_act_functions.insert(layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1

    """
    Function: network_remove_layer()
    Adds a layer at a specified position, with a given  number of units, init weight
    function, activation function.
    If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
    other layers are shifted.
    If the layer is inserted in position number_hidden_layers+1, then it is just appended
    to previous layer and it will output output_dim variables.
    If the position for the layer to be added is not within feasible bounds
    in the current architecture, the function silently returns
    """

    def network_remove_layer(self, layer_pos):

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it
          
        # We delete the layer in pos layer_pos     
        self.List_dims.pop(layer_pos) 
        self.List_init_functions.pop(layer_pos)   
        self.List_act_functions.pop(layer_pos)
         
        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        # print(layer_pos, "Old act fn ",self.List_act_functions[layer_pos], "New act fn", new_act_fn)
        self.List_act_functions[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_init_functions[layer_pos] = new_weight_fn

    def change_all_weight_init_fns(self, new_weight_fn):
        # If not within feasible bounds, return
        for layer_pos in range(self.number_hidden_layers):
            self.List_init_functions[layer_pos] = new_weight_fn

    def change_dimensions_in_layer(self, layer_pos, new_dim):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.List_dims[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.List_dims)
        print(identifier, ' Init:', self.List_init_functions)
        print(identifier, ' Act:', self.List_act_functions)
        print(identifier, ' Loop:', self.number_loop_train)

    def codify_components(self, max_hidden_layers, ref_list_init_functions, ref_list_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops      
        code = [self.number_hidden_layers, self.number_loop_train]
 
        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + self.List_dims + [-1]*(max_total_layers-len(self.List_dims))
 
        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.List_init_functions:
            aux_f.append(ref_list_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))
 
        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.List_act_functions:
            aux_a.append(ref_list_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))
 
        return code 

###############################################################################################################################
# ########################################################## Network #######################################################################################################################################################################################


class Network:
    def __init__(self, network_descriptor):
        self.descriptor = network_descriptor
        self.List_layers = []
        self.List_weights = []        
        self.List_bias = []
        self.List_dims = []
        self.List_init_functions = []
        self.List_act_functions = []

    def reset_network(self):  
        self.List_layers = []
        self.List_weights = []        
        self.List_bias = []   
        self.List_dims = []
        self.List_init_functions = []
        self.List_act_functions = []

    @staticmethod
    def create_hidden_layer(in_size, out_size, init_w_function, layer_name):
        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)
        return w, b
    
    def network_initialization(self):
        for lay in range(self.descriptor.number_hidden_layers+1):   
            init_w_function = self.descriptor.List_init_functions[lay]    
            if lay == 0:
                in_size = self.descriptor.input_dim
                out_size = self.descriptor.List_dims[lay]
            elif lay == self.descriptor.number_hidden_layers:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.output_dim 
            else:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.List_dims[lay]

            w, b = self.create_hidden_layer(in_size, out_size, init_w_function, str(lay))

            self.List_weights.append(w)        
            self.List_bias.append(b)

    def network_evaluation(self, layer):

        for lay in range(self.descriptor.number_hidden_layers+1):
            w = self.List_weights[lay]
            b = self.List_bias[lay] 
            act = self.descriptor.List_act_functions[lay]
            if act is None:
                layer = tf.matmul(layer, w) + b
            else:
                if act is not None:
                    layer = act(tf.matmul(layer, w) + b)
            self.List_layers.append(layer)    

        return layer


###############################################################################################################################
# ########################################################## GAN Descriptor  #######################################################################################################################################################################################

class GANDescriptor:
    def __init__(self, x_dim, z_dim, latent_distribution_function=np.random.uniform, fmeasure="Standard_Divergence"):
        self.X_dim = x_dim
        self.z_dim = z_dim
        self.latent_distribution_function = latent_distribution_function
        self.fmeasure = fmeasure
        self.Gen_network = []
        self.Disc_network = []

    def copy_from_other(self, other):
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.latent_distribution_function = other.latent_distribution_function

        self.fmeasure = other.fmeasure

        self.Gen_network = copy.deepcopy(other.Gen_network)     # These are  Network_Descriptor structures
        self.Disc_network = copy.deepcopy(other.Disc_network)

    def gan_generator_initialization(self, generator_n_hidden, generator_dim_list, generator_init_functions,
                                     generator_act_functions, generator_number_loop_train):

        self.Gen_network = NetworkDescriptor(generator_n_hidden, self.z_dim, self.X_dim, generator_dim_list,
                                             generator_init_functions, generator_act_functions, generator_number_loop_train)

    def gan_discriminator_initialization(self, discriminator_n_hidden, discriminator_dim_list, discriminator_init_functions,
                                         discriminator_act_functions, discriminator_number_loop_train):
        output_dim = 1     
        self.Disc_network = NetworkDescriptor(discriminator_n_hidden, self.X_dim, output_dim, discriminator_dim_list,
                                              discriminator_init_functions, discriminator_act_functions, discriminator_number_loop_train)

    def print_components(self):
        self.Gen_network.print_components("Gen")
        self.Disc_network.print_components("Disc")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:', self.fmeasure)

    def codify_components(self, max_layers, ref_list_init_functions, ref_list_act_functions, ref_list_divergence_functions, ref_list_latent_functions):

        latent_index = ref_list_latent_functions.index(self.latent_distribution_function)
        diverg_index = ref_list_divergence_functions.index(self.fmeasure)

        # The first two elements are the indices of the latent and divergence functions
        code = [latent_index, diverg_index]

        # Ve add the code of the generator
        code = code + self.Gen_network.codify_components(max_layers, ref_list_init_functions, ref_list_act_functions)

        # Ve add the code of the discriminator
        code = code + self.Disc_network.codify_components(max_layers, ref_list_init_functions, ref_list_act_functions)

        return code 


###################################################################################################
# ############################################ GAN  ################################################
###################################################################################################

class GAN:
    def __init__(self, gan_descriptor):
        self.descriptor = gan_descriptor
        self.Div_Function = self.Divergence_Functions[self.descriptor.fmeasure]
        self.Gen_network = None
        self.Disc_network = None
        self.G_sample = None
        self.X = None
        self.Z = None
        self.D_real = None
        self.D_logit_real = None
        self.D_fake = None
        self.D_logit_fake = None
        self.G_solver = None
        self.D_solver = None
        self.G_loss = 0
        self.D_loss = 0
        self.fmeasure = None

    def reset_network(self):
        self.Gen_network.reset_network()
        self.Disc_network.reset_network()

    def sample_z(self, m, n):
        return self.descriptor.latent_distribution_function(-1., 1., size=[m, n])

    def generator(self, z):
        g_log_prob = self.Gen_network.network_evaluation(z)
        g_prob = tf.nn.sigmoid(g_log_prob)
        return g_prob

    def discriminator(self, x):
        d_logit = self.Disc_network.network_evaluation(x)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

    def training_definition(self, lr):
        # =============================== TRAINING ====================================

        self.X = tf.placeholder(tf.float32, shape=[None, self.descriptor.X_dim], name="PlaceholderX")
        self.Z = tf.placeholder(tf.float32, shape=[None, self.descriptor.z_dim], name="PlaceholderZ")

        self.Gen_network = Network(self.descriptor.Gen_network)
        self.Disc_network = Network(self.descriptor.Disc_network)

        self.Gen_network.network_initialization()
        self.Disc_network.network_initialization()

        with tf.variable_scope('Gen1') as scope:
                self.G_sample = self.generator(self.Z)
        with tf.variable_scope('Disc1') as scope:
                self.D_real, self.D_logit_real = self.discriminator(self.X)
        with tf.variable_scope('Disc2') as scope:
                self.D_fake, self.D_logit_fake = self.discriminator(self.G_sample)

        self.Div_Function(self)

        self.G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=[self.Gen_network.List_weights, self.Gen_network.List_bias])
        self.D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=[self.Disc_network.List_weights, self.Disc_network.List_bias])

    def running(self, batch_type, data, mb_size, number_iterations, n_samples, print_cycle):
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('gan_out/'):
            os.makedirs('gan_out/')

        if batch_type == "random":
                    batch_function = next_random_batch
        else:
                    batch_function = next_batch

        i = 0
        for it in range(number_iterations):

            x_mb = batch_function(mb_size, data, i)
            i = i+mb_size if (i+mb_size) < len(data) else 0

            z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
            _, d_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={self.X: x_mb, self.Z: z_mb})
            _, g_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: z_mb})

            if it > 0 and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('D Loss: {:.4}'. format(d_loss_curr))
                print('G_Loss: {:.4}'. format(g_loss_curr))
                print()

        samples = sess.run(self.G_sample, feed_dict={self.Z: self.sample_z(n_samples, self.descriptor.z_dim)})

        return samples

    def separated_running(self, batch_type, data, mb_size, number_iterations, n_samples, print_cycle):
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('gan_out/'):
            os.makedirs('gan_out/')
        if batch_type == "random":
                    batch_function = next_random_batch
        else:
                    batch_function = next_batch

        i = 0
        for it in range(number_iterations):
            # Learning loop for discriminator

            d_loss_curr = 0
            g_loss_curr = 0
            for loop_disc in range(self.Gen_network.descriptor.number_loop_train):
                    z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
                    x_mb = batch_function(mb_size, data, i)
                    i = i+mb_size if (i+mb_size) < len(data) else 0
                    _, d_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={self.X: x_mb, self.Z: z_mb})

            # Learning loop for generator
            for loop_gen in range(self.Disc_network.descriptor.number_loop_train):
                    z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
                    i = i+mb_size if (i+mb_size) < len(data) else 0
                    _, g_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: z_mb})

            if it > 0 and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('D Loss: {:.4}'. format(d_loss_curr))
                print('G_Loss: {:.4}'. format(g_loss_curr))
                print()

        samples = sess.run(self.G_sample, feed_dict={self.Z: self.sample_z(n_samples, self.descriptor.z_dim)})

        return samples

    def standard_divergence(self):
            self.D_loss = -tf.reduce_mean(self.D_logit_real) + tf.reduce_mean(self.D_logit_fake)
            self.G_loss = -tf.reduce_mean(self.D_logit_fake)

    def total_variation(self):
            self.D_loss = -(tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_real)) - tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_fake)))
            self.G_loss = -tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_fake))

    def forward_kl(self):
            self.D_loss = -(tf.reduce_mean(self.D_logit_real) - tf.reduce_mean(tf.exp(self.D_logit_fake-1)))
            self.G_loss = -tf.reduce_mean(tf.exp(self.D_logit_fake-1))

    def reverse_kl(self):
            self.D_loss = -(tf.reduce_mean(-self.D_real) - tf.reduce_mean(1-self.D_logit_fake))
            self.G_loss = -tf.reduce_mean(-1 - self.D_logit_fake)

    def pearson_chi_squared(self):
            self.D_loss = -(tf.reduce_mean(self.D_logit_real) - tf.reduce_mean(0.25*self.D_logit_fake**2 + self.D_logit_fake))
            self.G_loss = -tf.reduce_mean(0.25*self.D_logit_fake**2 + self.D_logit_fake)

    def squared_hellinger(self):
            self.D_loss = -(tf.reduce_mean(1 - self.D_real) - tf.reduce_mean(self.D_fake / (1-self.D_fake)))
            self.G_loss = -tf.reduce_mean((1 - self.D_fake) / self.D_fake)

    def least_squared(self):
            self.D_loss = 0.5 * (tf.reduce_mean((self.D_logit_real - 1)**2) + tf.reduce_mean(self.D_logit_fake**2))
            self.G_loss = 0.5 * tf.reduce_mean((self.D_logit_fake - 1)**2)

    Divergence_Functions = {"Standard_Divergence": standard_divergence,
                            "Total_Variation": total_variation,
                            "Forward_KL": forward_kl,
                            "Reverse_KL": reverse_kl,
                            "Pearson_Chi_squared": pearson_chi_squared,
                            "Squared_Hellinger": squared_hellinger,
                            "Least_squared": least_squared}

    def set_divergence_function(self, fmeasure):
        self.fmeasure = fmeasure
        self.Div_Function = self.Divergence_Functions[self.fmeasure]


###############################################################################################################################
# ########################################################## VAE Descriptor  #######################################################################################################################################################################################


class VAEESDescriptor:
    def __init__(self, x_dim, z_dim, objectives, latent_distribution_function,  fmeasure):
        self.X_dim = x_dim
        self.z_dim = z_dim
        self.objectives = objectives
        # latent_distribution_function is left here for consistency with GAN and future extensions
        # but currently, only Normal distribution is used for Z (think about of flow distributions)
        self.latent_distribution_function = latent_distribution_function
        self.fmeasure = fmeasure
        self.Enc_network = None
        self.Dec_network = None
        self.App_network = None

    def copy_from_other(self, other):
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.latent_distribution_function = other.latent_distribution_function

        self.fmeasure = other.fmeasure

        self.Enc_network = other.Gen_network     # These are  Network_Descriptor structures
        self.Dec_network = other.Disc_network
        self.App_network = other.App_network

    def vae_encoder_initialization(self, encoder_n_hidden, encoder_dim_list, encoder_init_functions,
                                   encoder_act_functions, encoder_number_loop_train=1):

        input_dim = self.X_dim
        output_dim = self.z_dim
        self.Enc_network = NetworkDescriptor(encoder_n_hidden, input_dim, output_dim, encoder_dim_list,
                                             encoder_init_functions, encoder_act_functions, encoder_number_loop_train)

    def vae_decoder_initialization(self, decoder_n_hidden, decoder_dim_list, decoder_init_functions,
                                   decoder_act_functions, decoder_number_loop_train=1):
        output_dim = self.X_dim
        input_dim = self.z_dim + self.objectives

        self.Dec_network = NetworkDescriptor(decoder_n_hidden, input_dim, output_dim, decoder_dim_list, decoder_init_functions,
                                             decoder_act_functions, decoder_number_loop_train)

    def vae_approximator_initialization(self, approximator_n_hidden, approximator_dim_list, approximator_init_functions,
                                        approximator_act_functions, approximator_number_loop_train=1):

        input_dim = self.z_dim + self.objectives
        output_dim = self.objectives
        self.App_network = NetworkDescriptor(approximator_n_hidden, input_dim, output_dim, approximator_dim_list,
                                             approximator_init_functions, approximator_act_functions, approximator_number_loop_train)

    def print_components(self):
        self.Enc_network.print_components("Enc")
        self.Dec_network.print_components("Dec")
        self.App_network.print_components("App")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:',  self.fmeasure)

    def codify_components(self, max_layers, ref_list_init_functions, ref_list_act_functions, ref_list_divergence_functions, ref_list_latent_functions):

        latent_index = ref_list_latent_functions.index(self.latent_distribution_function)
        diverg_index = ref_list_divergence_functions.index(self.fmeasure)

        # The first two elements are the indices of the latent and divergence functions
        code = [latent_index, diverg_index]

        # Ve add the code of the discriminator
        code = code + self.Dec_network.codify_components(max_layers, ref_list_init_functions, ref_list_act_functions)

        # Ve add the code of the discriminator
        code = code + self.Enc_network.codify_components(max_layers, ref_list_init_functions, ref_list_act_functions)

        # Ve add the code of the generator
        code = code + self.App_network.codify_components(max_layers, ref_list_init_functions, ref_list_act_functions)

        return code


class VAE(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, _, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Div_Function = self.Divergence_Functions[self.network_architecture.fmeasure]

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture.X_dim], name="PlaceholderX")
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

    def reinitialize_net(self):
        self.sess.run(tf.global_variables_initializer())

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
        x_reconstr_mean = tf.nn.sigmoid(self.gen_network.network_evaluation(self.z))
        return x_reconstr_mean

    def _create_loss_optimizer(self):

        self.Div_Function(self)
        self.latent_loss = -0.0005 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def vae_mse_div(self):
        # E[log P(X|z)]
        self.reconstr_loss = tf.losses.mean_squared_error(predictions=self.x_reconstr_mean, labels=self.x)

    def vae_log_prob(self):
        self.reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1 - self.x_reconstr_mean), 1)

    Divergence_Functions = {"VAE_Log_Prob": vae_log_prob, "VAE_MSE_Div": vae_mse_div}

    def partial_fit(self, x, _):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        _, cost1, cost2 = self.sess.run((self.optimizer, self.reconstr_loss, self.latent_loss), feed_dict={self.x: x})

        return np.sum(cost1), np.sum(cost2)

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
        samples = self.sess.run(self.x_reconstr_mean, feed_dict={self.z: np.reshape(z_mu, (samples, -1))})
        #print("Samples")
        #print(np.min(samples), np.max(samples))

        return samples

    def reconstruct(self, x):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: x})

    def train(self, data, objs, batch_size=10, epochs=10, display_step=100):

        # Training cycle
        batch_i = 0
        n_samples = data.shape[0]
        #print("Data")
        #print(np.min(data), np.max(data))
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for _ in range(total_batch):
                batch_i, batch_xs, batch_ys = batch(data, objs, batch_size, batch_i)
                # Fit training using batch data
                # print(self.sess.run(self.x_reconstr_mean, feed_dict={self.x: batch_xs, self.obj: batch_ys}))
                cost = self.partial_fit(batch_xs, batch_ys)
                # Compute average loss
                #avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0 and False:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=" + str(cost))


class VAEE(VAE):

    def __init__(self, network_architecture, objectives):

        self.objectives = objectives
        self.approximation = None
        self.obj = tf.placeholder(tf.float32, shape=[None, self.objectives], name="PlaceholderObjs")

        super().__init__(network_architecture, objectives)

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

        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.app_cost = tf.losses.mean_squared_error(predictions=self.approximation, labels=self.obj)

        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss + self.app_cost)

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

        _, cost1, cost2, cost3 = self.sess.run((self.optimizer, self.reconstr_loss, self.app_cost, self.latent_loss), feed_dict={self.x: x, self.obj: obj})

        return np.sum(cost1), cost2, np.sum(cost3)

    def generate(self, samples, objs):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """

        z_mu = np.random.normal(size=(samples, self.network_architecture.z_dim))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        samples = self.sess.run([self.x_reconstr_mean, self.approximation], feed_dict={self.z: np.reshape(z_mu, (samples, -1)), self.obj: objs})

        return samples


class VAEEE(VAEE):

    def __init__(self, network_architecture, objectives):
        super().__init__(network_architecture, objectives)

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        gen_descriptor = self.network_architecture.Dec_network
        self.gen_network = Network(gen_descriptor)
        self.gen_network.network_initialization()
        x_reconstr_mean = tf.nn.sigmoid(self.gen_network.network_evaluation(tf.concat((self.z, self.obj), axis=1)))
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

