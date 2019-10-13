import tensorflow as tf

from functions.functions import tf_batch_matmul


def encoder(x, k, name, units_list=[256, 256, 128]):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        h = x
        for i, units in enumerate(units_list):
            h = tf.layers.dense(
                inputs=h,
                units=units,
                activation=tf.nn.tanh,
                name="dense_%i" % i
            )
        mu = tf.layers.dense(
            inputs=h,
            units=k,
            name="mu"
        )
        log_sigma = tf.layers.dense(
            inputs=h,
            units=k,
            name="log_sigma"
        )

    return mu, log_sigma


def decoder(z, k, x_depth, name, units_list=[128, 256, 256]):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        h = z
        for i, units in enumerate(units_list):
            h = tf.layers.dense(
                inputs=h,
                units=units,
                activation=tf.nn.tanh,
                name="dense_%i" % i
            )
        f = tf.layers.dense(
            inputs=h,
            units=x_depth,
            activation=tf.nn.sigmoid,
            name="f"
        )
    
    return f


class VAE(object):
    def __init__(self, x, eps, k, x_depth):
        self.x = x
        self.eps = eps
        self.k = k
        self.x_depth = x_depth

        self.mu, self.log_sigma = encoder(self.x, self.k, name="encoder")
        self.sigma = tf.exp(self.log_sigma)

        self.z = tf.sqrt(self.sigma) * self.eps + self.mu

        self.f_z = decoder(self.z, self.k, self.x_depth, name="decoder")
        self.f_eps = decoder(self.eps, self.k, self.x_depth, name="decoder")
    
    def get_encoder_loss(self):
        loss = \
            (1/2) * (tf.reduce_sum(self.sigma, axis=-1, keepdims=True) + \
                tf.reduce_sum(self.mu * self.mu, axis=-1, keepdims=True) - \
                    self.k - \
                        tf.reduce_sum(self.log_sigma, axis=-1, keepdims=True)
            )
        
        return loss
    
    def get_decoder_loss(self):
        loss = tf.reduce_sum((self.x - self.f_z)**2, axis=-1, keepdims=True)

        return loss