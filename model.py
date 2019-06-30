'''
# Beta Variational AutoEncoder Model
# Author: Yuki Saeki
# Reference: "https://github.com/Silver-L/beta-VAE"
'''

import tensorflow as tf
import os
import numpy as np

class Variational_Autoencoder(object):

    def __init__(self, sess, outdir, beta, latent_dim, batch_size, image_size, encoder, decoder, is_res, is_constraints=False, keep_prob=1, learning_rate=1e-5):

        self._sess = sess
        self._outdir = outdir
        self._beta = beta
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._image_size = image_size
        self._learning_rate = learning_rate
        self._encoder = encoder
        self._decoder = decoder
        self._res = is_res
        self._constraints = is_constraints
        self._keep_prob = keep_prob

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('input'):
            if len(self._image_size) == 1:
                self.input = tf.placeholder(tf.float32, shape = [None, 9*9*9])
            if self._res is True:
                self.input2 = tf.placeholder(tf.float32, shape = [None, 9*9*9])
                self.input = self.input - self.input2

            if len(self._image_size) == 3:
                self.input = tf.placeholder(tf.float32, shape = [None, self._image_size[0], self._image_size[1],
                                                                 self._image_size[2]])


        with tf.variable_scope('encoder'):
            encoded, self._shape_size = self._encoder(self.input, self._latent_dim, self._keep_prob)
            # tf.summary.histogram("encoder", encoded)      #histogram

        with tf.variable_scope('latent_space'):
            with tf.variable_scope('latent_mean'):
                self.mean = encoded[:, :self._latent_dim]

            with tf.variable_scope('latent_sigma'):
                stddev = encoded[:, self._latent_dim:]
                # stddev = tf.sqrt(tf.exp(logvar))

            epsilon = tf.random_normal([self._batch_size, self._latent_dim], mean=0., stddev=1.0)
            with tf.variable_scope('latent_sample'):
                self.z = self.mean + stddev * epsilon

        with tf.variable_scope('decoder'):
            self.decode_result = self._decoder(self.z, self._batch_size, self._shape_size, self._keep_prob)
            if self._res is True:
                self.decode_result = self.input2 + self.decode_result
            if self._constraints is True:
                self.decode_result = tf.layers.dense(self.decode_result, np.prod(self._image_size), activation=tf.nn.sigmoid, name='final_sigmoid')

            # tf.summary.histogram("decoder", decoded)      #histogram

        with tf.variable_scope('loss'):
            with tf.variable_scope('reconstruction-loss'):
                rec_loss = self._rec_loss(self.decode_result, self.input, feature_size=np.prod(self._image_size))
                self._rec_loss = tf.reduce_mean(rec_loss)

            with tf.variable_scope('kl-divergence'):
                kl = self._kl_diagnormal_stdnormal(self.mean, stddev)
                self._kl_loss = tf.reduce_mean(kl)
                self._kl_loss *= self._beta

            self._loss = self._rec_loss + self._kl_loss
        self._val_loss = self._loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.9, beta2=0.999)

        with tf.variable_scope('training-step'):
            self._train = optimizer.minimize(self._loss)

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)


    @staticmethod
    def _kl_diagnormal_stdnormal(mu, z_sigma):
        kl = - 0.5 * tf.reduce_sum(1 + tf.log(tf.square(z_sigma) + 1e-5) - tf.square(mu) - tf.square(z_sigma), axis=-1)

        return kl

    @staticmethod
    def _rec_loss(pred, ground_truth, feature_size):
        pred_reshaped = tf.reshape(pred, [-1, feature_size])
        ground_truth_reshaped = tf.reshape(ground_truth, [-1, feature_size])
        se = tf.losses.mean_squared_error(pred_reshaped, ground_truth_reshaped) * feature_size
        return se

    def update(self, X):
        _, loss, loss_rec, loss_kl = self._sess.run([self._train, self._loss, self._rec_loss, self._kl_loss], feed_dict = {self.input : X})
        return loss, loss_rec, loss_kl

    def update2(self, X, Y):
        _, loss, loss_rec, loss_kl = self._sess.run([self._train, self._loss, self._rec_loss, self._kl_loss], feed_dict = {self.input : X, self.input2 : Y})
        return loss, loss_rec, loss_kl


    def save_model(self, index):
        save = self.saver.save(self._sess, os.path.join(self._outdir, 'model' , 'model_{}'.format(index)))
        return save

    def save_model2(self, index):
        save = self.saver.save(self._sess, os.path.join(self._outdir, 'model2' , 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)

    def validation(self, X):
        val_loss = self._sess.run(self._val_loss, feed_dict = {self.input : X})
        return val_loss

    def validation2(self, X, Y):
        val_loss = self._sess.run(self._val_loss, feed_dict = {self.input : X, self.input2: Y})
        return val_loss

    def reconstruction_image(self, X):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X})
        predict = self._sess.run(self.decode_result, feed_dict = {self.z: hidden})
        return predict

    def reconstruction_image2(self, X, Y):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X, self.input2: Y})
        predict = self._sess.run(self.decode_result, feed_dict = {self.z: hidden, self.input2: Y})
        return predict

    def plot_latent(self, X):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X})
        return hidden

    def plot_latent2(self, X, Y):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X, self.input2: Y})
        return hidden

    def generate_sample(self,z):
        x = self._sess.run(self.decode_result, feed_dict = {self.z: z})
        return  x

    def generate_sample2(self,z, Y):
        x = self._sess.run(self.decode_result, feed_dict = {self.z: z, self.input2: Y})
        return  x
