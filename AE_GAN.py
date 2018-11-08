#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from glob import glob

from ops import *
from utils import *

class AE_GAN(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "AE_GAN"     # name for checkpoint

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 1 #color channels

            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            #train iter
            self.train_iter = 0

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            out = linear(net, self.z_dim, scope='en_fc4')
            return out

    def decoder(self, z, is_training=True, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
                   scope='de_bn3'))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='de_dc4'))
            return out

    def generatorA(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generatorA", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 512, scope='gA_fc1'), is_training=is_training, scope='gA_bn1'))
            net = tf.nn.relu(bn(linear(net, 1024, scope='gA_fc2'), is_training=is_training, scope='gA_bn2'))
            net = tf.nn.relu(bn(linear(net, 512, scope='gA_fc3'), is_training=is_training, scope='gA_bn3'))
            out = linear(net, self.z_dim, scope='gA_fc4')
            return out

    def discriminatorA(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminatorA", reuse=reuse):
            net = tf.nn.relu(bn(linear(x, 512, scope='dA_fc1'), is_training=is_training, scope='dA_bn1'))
            net = tf.nn.relu(bn(linear(net, 1024, scope='dA_fc2'), is_training=is_training, scope='dA_bn2'))
            net = tf.nn.relu(bn(linear(net, 512, scope='dA_fc3'), is_training=is_training, scope='dA_bn3'))
            out_logit = linear(net, 1, scope='dA_fc4')
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

    def generatorB(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generatorB", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='gB_fc1'), is_training=is_training, scope='gB_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='gB_fc2'), is_training=is_training, scope='gB_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='gB_dc3'), is_training=is_training,
                   scope='gB_bn3'))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='gB_dc4'))
            return out

    def discriminatorB(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminatorB", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='dB_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='dB_conv2'), is_training=is_training, scope='dB_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = MinibatchLayer(32, 32, net, 'dB_fc3')
            net = lrelu(bn(linear(net, 1024, scope='dB_fc4'), is_training=is_training, scope='dB_bn4'))
            out_logit = linear(net, 1, scope='dB_fc5')
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

    def mse_loss(self, pred, data):
        # tf.nn.l2_loss(pred - data) = sum((pred - data) ** 2) / 2
        loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.batch_size
        return loss_val

    def build_model(self):
        # some parameters
        image_dims = [self.output_height, self.output_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function of z"""
        self.G_A_z = self.generatorA(self.z, is_training=True, reuse=False)
        self.En_z = self.encoder(self.inputs, is_training=True, reuse=False)

        D_encoder, D_encoder_logits = self.discriminatorA(self.En_z, is_training=True, reuse=False)
        D_fake_z, D_fake_z_logits = self.discriminatorA(self.G_A_z, is_training=True, reuse=True)

        # The AEGAN loss use LSGAN idea or you can use original GAN loss.

        """DiscriminatorA loss"""
        # dA_loss_encoder = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_encoder_logits, labels=tf.ones_like(D_encoder)))
        # dA_loss_fake_z = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_z_logits, labels=tf.zeros_like(D_fake_z)))

        dA_loss_encoder = tf.reduce_mean(self.mse_loss(D_encoder_logits, tf.ones_like(D_encoder_logits)))
        dA_loss_fake_z = tf.reduce_mean(self.mse_loss(D_fake_z_logits, tf.zeros_like(D_fake_z_logits)))
        self.dA_z_loss = dA_loss_encoder + dA_loss_fake_z

        """GeneratorA loss"""
        # self.gA_z_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_z_logits, labels=tf.ones_like(D_fake_z)))
        self.gA_z_loss = tf.reduce_mean(self.mse_loss(D_fake_z_logits, tf.ones_like(D_fake_z_logits)))

        """ Loss Function of x"""
        G_B_fake = self.generatorB(self.G_A_z, is_training=True, reuse=False)
        De_fake = self.decoder(self.En_z, is_training=True, reuse=False)

        """DiscriminatorB loss"""
        D_real, D_real_logits = self.discriminatorB(self.inputs, is_training=True, reuse=False)
        D_fake_G, D_fake_G_logits = self.discriminatorB(G_B_fake, is_training=True, reuse=True)
        D_fake_De, D_fake_De_logits = self.discriminatorB(De_fake, is_training=True, reuse=True)

        # dB_loss_decoder = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_De_logits, labels=tf.zeros_like(D_fake_De)))
        # dB_loss_fake_z = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_G_logits, labels=tf.zeros_like(D_fake_G)))
        # dB_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))

        dB_loss_decoder = tf.reduce_mean(self.mse_loss(D_fake_De_logits, tf.zeros_like(D_fake_De_logits)))
        dB_loss_fake_z = tf.reduce_mean(self.mse_loss(D_fake_G_logits, tf.zeros_like(D_fake_G_logits)))
        dB_loss_real = tf.reduce_mean(self.mse_loss(D_real_logits, tf.ones_like(D_real_logits)))
        self.dB_x_loss = dB_loss_fake_z + dB_loss_real + 1e-5*dB_loss_decoder

        """GeneratorB loss"""
        # gB_loss_decoder = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_De_logits, labels=tf.ones_like(D_fake_De)))
        # gB_loss_fake_z = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_G_logits, labels=tf.ones_like(D_fake_G)))
        gB_loss_decoder = tf.reduce_mean(self.mse_loss(D_fake_De_logits, tf.ones_like(D_fake_De_logits)))
        gB_loss_fake_z = tf.reduce_mean(self.mse_loss(D_fake_G_logits, tf.ones_like(D_fake_G_logits)))
        self.gB_x_loss = gB_loss_fake_z + 1e-5*gB_loss_decoder

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        dA_vars = [var for var in t_vars if 'dA_' in var.name]
        gA_vars = [var for var in t_vars if 'gA_' in var.name]
        dB_vars = [var for var in t_vars if 'dB_' in var.name]
        gB_vars = [var for var in t_vars if ('gB_' in var.name) or ('en_' in var.name) or ('de_' in var.name)]
        # gB_vars = [var for var in t_vars if 'gB_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.dA_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.dA_z_loss, var_list=dA_vars)
            self.gA_optim = tf.train.AdamOptimizer(self.learning_rate*8, beta1=self.beta1) \
                .minimize(self.gA_z_loss, var_list=gA_vars)
            self.dB_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.dB_x_loss, var_list=dB_vars)
            self.gB_optim = tf.train.AdamOptimizer(self.learning_rate*15, beta1=self.beta1) \
                .minimize(self.gB_x_loss, var_list=gB_vars)

        self.clip_DA = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dA_vars]
        self.clip_DB = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dB_vars]

        self.fake_images = self.generatorB(self.generatorA(self.z, is_training=False, reuse=True),
                                           is_training=False, reuse=True)

        """ Summary """
        dA_loss_encoder_sum = tf.summary.scalar("dA_loss_encoder", dA_loss_encoder)
        dA_loss_fake_z_sum = tf.summary.scalar("dA_loss_fake_z", dA_loss_fake_z)
        dA_z_loss_sum = tf.summary.scalar("dA_z_loss_sum", self.dA_z_loss)
        dB_loss_decoder_sum = tf.summary.scalar("dB_loss_decoder", dB_loss_decoder)
        dB_loss_fake_z_sum = tf.summary.scalar("dB_loss_fake_z", dB_loss_fake_z)
        dB_loss_real_sum = tf.summary.scalar("dB_loss_real", dB_loss_real)
        dB_x_loss_sum = tf.summary.scalar("dB_x_loss_sum", self.dB_x_loss)
        gA_loss_sum = tf.summary.scalar("gA_loss", self.gA_z_loss)
        gB_loss_sum = tf.summary.scalar("gB_loss", self.gB_x_loss)

        # final summary operations
        self.gA_sum = tf.summary.merge([dA_loss_fake_z_sum, gA_loss_sum])
        self.dA_sum = tf.summary.merge([dA_loss_encoder_sum, dA_z_loss_sum])
        self.gB_sum = tf.summary.merge([dB_loss_fake_z_sum, dB_loss_decoder_sum, gB_loss_sum])
        self.dB_sum = tf.summary.merge([dB_loss_real_sum, dB_x_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        if start_epoch != self.epoch:
            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]

                # update DA network
                _, summary_str, dA_loss = self.sess.run([self.dA_optim, self.dA_sum, self.dA_z_loss],
                                                        feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update GA network
                _, summary_str, gA_loss = self.sess.run([self.gA_optim, self.gA_sum, self.gA_z_loss],
                                                        feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update DB network
                _, summary_str, dB_loss = self.sess.run([self.dB_optim, self.dB_sum, self.dB_x_loss],
                                                        feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update GB, De, En network
                _, summary_str, gB_loss = self.sess.run([self.gB_optim, self.gB_sum, self.gB_x_loss],
                                                        feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                # display training status
                counter += 1

                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, dA_loss: %.8f, gA_loss: %.8f,dB_loss: %.8f, gB_loss: "
                          "%.8f"\
                          % (epoch, idx, self.num_batches, time.time() - start_time, dA_loss, gA_loss, dB_loss, gB_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)
        generate_animation(self.result_dir + '/' + self.model_dir + '/' + self.model_name, self.epoch)
        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch
                    + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")



