import os
import pathlib
import re
import time

import numpy as np
import rasterio.fill

import imageio
import scipy.misc
import tensorflow as tf
from ops_alex import *
from ops_sn import *


class DCGAN(object):
    def __init__(self, sess,
                 batch_size=16, sample_size=128, gf_dim=64, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.input_size = sample_size
        self.df_dim = df_dim
        self.gf_dim = gf_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.c_bn0 = batch_norm(is_train, name='c_bn0')
        self.c_bn1 = batch_norm(is_train, name='c_bn1')
        self.c_bn2 = batch_norm(is_train, name='c_bn2')
        self.c_bn3 = batch_norm(is_train, name='c_bn3')

        self.build_model(is_train)

    def build_model(self, is_train):

        self.img_next_gt = tf.placeholder(tf.float32, shape=(
            self.batch_size, self.sample_size, self.sample_size, 3), name='img_next_gt')

        self.img_gt = tf.placeholder(tf.float32, shape=(
            self.batch_size, self.sample_size, self.sample_size, 1), name='img_gt')

        with tf.variable_scope('generator') as scope:
            self.img_out = self.encoder(self.img_next_gt)

        with tf.variable_scope('discriminator') as scope:
            d_fake_local = self.discriminator(
                self.img_out, update_collection=None)
            d_real_local = self.discriminator(
                self.img_gt, reuse=tf.AUTO_REUSE, update_collection="NO_OPS")

            self.real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_real_local), logits=d_real_local))
            self.fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(d_fake_local), logits=d_fake_local))
            self.d_loss = self.real_loss + self.fake_loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_fake_local), logits=d_fake_local))

        with tf.variable_scope('L2_loss') as scope:
            self.loss = tf.reduce_mean(
                tf.square(self.img_out - self.img_gt))

        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()

        # define variables to train in optimizer
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        # save variables
        self.saver = tf.train.Saver(self.g_vars + self.d_vars +
                                    batch_norm.shadow_variables,
                                    max_to_keep=0)

    def train(self, config, run_string="???"):
        """Train DCGAN"""

        # start from chekpoint if there exist
        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)

        # Learning rate of generator is gradually decreasing.
        self.g_lr = tf.train.exponential_decay(
            0.0002, global_step=global_step, decay_steps=20000, decay_rate=0.9, staircase=True)

        # define optimizer
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=config.beta1) \
                          .minimize(20 * self.loss + self.g_loss, var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        # # # See that moving average is also updated with g_optim.
        with tf.control_dependencies([g_optim]):
            g_optim = tf.group(self.bn_assigners)

        # initializer.Don't understand, just leave it there
        tf.global_variables_initializer().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(
                config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        # save the summary to check in tensorboard
        self.make_summary_ops()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            config.summary_dir, graph_def=self.sess.graph_def)

        # iterate over dataset
        data_root = pathlib.Path('/mnt/data/banana_data')
        ids = [p.stem for p in data_root.joinpath('height').glob('*.tif')]

        train, test = [], []
        for index, rand in enumerate(np.random.rand(len(ids))):
            if rand < 0.9:
                train.append(ids[index])
            else:
                test.append(ids[index])

        kernel = np.zeros((8, 8))
        kernel[0, 0] = 1
        mask = np.tile(kernel, (128, 128))

        s1 = 512  # scale 1
        s2 = 256  # scale 2
        s3 = 128  # scale 3

        input_size = self.input_size
        loss_all = []

        for _ in range(200):
            for id in train:
                counter += 1

                image_next_gt = imageio.imread(
                    data_root.joinpath('image', id + '.png'))

                img_tmp = []
                img_next_lst_s2 = []
                img_next_lst_s3 = []
                img_tmp.append(image_next_gt[:s1, :s1, :])
                img_tmp.append(image_next_gt[:s1, s1:, :])
                img_tmp.append(image_next_gt[s1:, :s1, :])
                img_tmp.append(image_next_gt[s1:, s1:, :])

                for ii in range(len(img_tmp)):
                    for r in range(2):
                        for c in range(2):
                            tmp = img_tmp[ii][s2 * r:s2 *
                                              (r + 1), s2 * c:s2 * (c + 1)]
                            img_next_lst_s2.append(scipy.misc.imresize(
                                tmp, [input_size, input_size]))

                img_next_gt = np.array(img_next_lst_s2)
                img_next_gt = img_next_gt / 255.0 * 2 - 1  # nomalize

                image_gt = imageio.imread(
                    data_root.joinpath('height', id + '.tif'))
                image_gt = rasterio.fill.fillnodata(
                    image_gt, mask=mask)  # interpolate

                # skip tiles without any building parts
                if np.sum(image_gt) == 0:
                    # print('no building: ' + id)
                    continue

                img_tmp = []
                img_next_lst_s2 = []
                img_next_lst_s3 = []
                img_tmp.append(image_gt[:s1, :s1])
                img_tmp.append(image_gt[:s1, s1:])
                img_tmp.append(image_gt[s1:, :s1])
                img_tmp.append(image_gt[s1:, s1:])

                for ii in range(len(img_tmp)):
                    for r in range(2):
                        for c in range(2):
                            tmp = img_tmp[ii][s2 * r:s2 *
                                              (r + 1), s2 * c:s2 * (c + 1)]
                            img_next_lst_s2.append(scipy.misc.imresize(
                                tmp, [input_size, input_size]).reshape(input_size, input_size, 1))

                img_gt = np.array(img_next_lst_s2)
                img_gt = img_gt / 255.0 * 2 - 1  # nomalize

                # sanity check
                if np.isnan(img_gt).any() or np.isnan(img_next_gt).any() or np.isinf(img_gt).any() or np.isinf(img_next_gt).any():
                    print('Dirty pic: ', id)
                    continue

                # after process the numpy, map the numpy variable to tensor variable
                feed_dict = {self.img_next_gt: img_next_gt,
                             self.img_gt: img_gt}

                # run the session, which is already opened in main.py. Give the feed_dict
                _, _, loss, g_loss, real_loss, fake_loss = self.sess.run(
                    [g_optim, d_optim, self.loss, self.g_loss, self.real_loss, self.fake_loss], feed_dict=feed_dict)
                image_out, image_out_gt, image_in = self.sess.run(
                    [self.img_out, self.img_gt, self.img_next_gt], feed_dict=feed_dict)
                #print('l2: ' + str(loss), 'g: ' + str(g_loss), 'real: ' + str(real_loss), 'fake: ' + str(fake_loss))
                if np.isfinite(np.nanmax(image_gt)):
                    loss_all.append(np.sqrt(loss) * np.nanmax(image_gt))

                if np.mod(counter, 50) == 1:
                    print('Counter: ' + str(counter))
                    print('Mean loss: ' + str(np.mean(np.array(loss_all))))
                    self.save(config.checkpoint_dir, counter)

                    image_out = (image_out[0].reshape(
                        input_size, input_size) + 1) / 2 * 255
                    image_in = (image_in[0] + 1) / 2 * 255
                    image_out_gt = (image_out_gt[0] + 1) / 2 * 255

                    imageio.imsave(os.path.join(
                        config.summary_dir, '%s_in.png' % id), image_in)
                    imageio.imsave(os.path.join(
                        config.summary_dir, '%s_out.png' % id), image_out)
                    imageio.imsave(os.path.join(
                        config.summary_dir, '%s_out_gt.png' % id), image_out_gt)

    def discriminator(self, image1, reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            c0_0 = sn_lrelu(sn_conv2d(image1,  64, 3, 3, 1, 1, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c0_0'))
            c0_1 = sn_lrelu(sn_conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c0_1'))
            c1_0 = sn_lrelu(sn_conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c1_0'))
            c1_1 = sn_lrelu(sn_conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c1_1'))
            c2_0 = sn_lrelu(sn_conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c2_0'))
            c2_1 = sn_lrelu(sn_conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c2_1'))
            c3_0 = sn_lrelu(sn_conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True,
                                      update_collection=update_collection, stddev=0.02, name='d_c3_0'))
            c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
            l4 = sn_linear(c3_0, 1, spectral_normed=True,
                           update_collection=update_collection, stddev=0.02, name='d_l4')
            return tf.reshape(l4, [-1])

    def encoder(self, image, reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            conv1 = lrelu(self.c_bn0(
                conv2d(image, self.df_dim, k_w=3, k_h=3, name='g_conv1')))
            conv2 = lrelu(self.c_bn1(
                conv2d(conv1, self.df_dim * 2, k_w=3, k_h=3, name='g_conv2')))
            conv3 = lrelu(self.c_bn2(
                conv2d(conv2, self.df_dim * 4, k_w=3, k_h=3, name='g_conv3')))
            conv4 = lrelu(self.c_bn3(
                conv2d(conv3, self.df_dim * 8, k_w=3, k_h=3, name='g_conv4')))

            dc = deconv2d(conv4, [self.batch_size, 16, 16, self.df_dim * 8],
                          k_h=4, k_w=4, name='g_h0')
            dc = tf.nn.relu(instance_norm(dc))
            dc = deconv2d(dc, [self.batch_size, 32, 32, self.df_dim * 4],
                          k_h=4, k_w=4, name='g_h1')
            dc = tf.nn.relu(instance_norm(dc))
            dc = deconv2d(dc, [self.batch_size, 64, 64, self.df_dim * 2],
                          k_h=4, k_w=4, name='g_h2')
            dc = tf.nn.relu(instance_norm(dc))
            dc = deconv2d(dc, [self.batch_size, 128, 128, self.df_dim * 1],
                          k_h=4, k_w=4, name='g_h3')
            dc = tf.nn.relu(instance_norm(dc))
            dc = deconv2d(dc, [self.batch_size, 128, 128, 1], k_h=4,
                          k_w=4, d_h=1, d_w=1, name='g_h4')
            return tf.nn.tanh(dc)

    # Function to save into tensorboard to visualize the loss change
    def make_summary_ops(self):
        tf.summary.scalar('g_loss', self.loss)

    # save checkpoint
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name
