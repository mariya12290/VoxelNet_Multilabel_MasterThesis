#!/usr/bin/env python
# -*- coding:UTF-8 -*-


import tensorflow as tf
import numpy as np

from config import cfg


small_addon_for_BCE = 1e-6


class MiddleAndRPN:
    def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        self.input = input  #feature network  
        self.training = training
        # groundtruth(target) - each anchor box, represent as △x, △y, △z, △l, △w, △h, rotation
        #Pedestrian
        self.targets_p = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 14])
        # postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        self.pos_equal_one_p = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 2])
        self.pos_equal_one_sum_p = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg_p = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 14])
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one_p = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 2])
        self.neg_equal_one_sum_p = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 1, 1])
        
        #cyclist
        #groundtruth(target) - each anchor box, represent as △x, △y, △z, △l, △w, △h, rotation
        self.targets_c = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 14])
        # postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        self.pos_equal_one_c = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 2])
        self.pos_equal_one_sum_c = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg_c = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 14])
        #negative anchors equal to one and others equal to zero
        self.neg_equal_one_c = tf.compat.v1.placeholder(
            tf.compat.v1.float32, [None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 2])
        self.neg_equal_one_sum_c = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 1, 1])


        with tf.compat.v1.variable_scope('MiddleAndRPN_' + name):
            # convolutinal middle layers
            temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1),
                               (1, 1, 1), self.input, name='conv1')
            temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1),
                               (0, 1, 1), temp_conv, name='conv2')
            temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1),
                               (1, 1, 1), temp_conv, name='conv3')
            temp_conv = tf.compat.v1.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
            temp_conv = tf.compat.v1.reshape(
                temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])

            # rpn
            # block1:
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv4')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv5')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv6')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv7')
            deconv1 = Deconv2D(128, 256, 3, (1, 1), (0, 0),
                               temp_conv, training=self.training, name='deconv1')

            # block2:
            temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv8')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv9')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv10')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv11')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv12')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv13')
            deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0),
                               temp_conv, training=self.training, name='deconv2')

            # block3:
            temp_conv = ConvMD(2, 128, 256, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv14')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv15')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv16')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv17')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv18')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv19')
            deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0),
                               temp_conv, training=self.training, name='deconv3')

            # final:
            temp_conv = tf.compat.v1.concat([deconv3, deconv2, deconv1], -1)
            #Pedestrian

            # Probability score map, scale = [None, 200/100, 176/120, 2]
            p_map_p = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv,
                           training=self.training, activation=False, bn=False, name='convp20')  #make output channel 4
            # Regression(residual) map, scale = [None, 200/100, 176/120, 14]
            r_map_p = ConvMD(2, 768, 14, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='convp21')  #make output channel 28
            # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1]
            self.p_pos_p = tf.compat.v1.sigmoid(p_map_p,name='probp') # 
            #self.p_pos = tf.compat.v1.nn.softmax(p_map, dim=3)

            self.cls_pos_loss_p = (-self.pos_equal_one_p * tf.compat.v1.log(self.p_pos_p + small_addon_for_BCE)) / self.pos_equal_one_sum_p
        
            self.cls_neg_loss_p = (-self.neg_equal_one_p * tf.compat.v1.log(1 - self.p_pos_p + small_addon_for_BCE)) / self.neg_equal_one_sum_p
            
            self.cls_pos_loss_rec_p = tf.compat.v1.reduce_sum( self.cls_pos_loss_p )
            self.cls_neg_loss_rec_p = tf.compat.v1.reduce_sum( self.cls_neg_loss_p )
                
            self.reg_loss_p = smooth_l1(r_map_p * self.pos_equal_one_for_reg_p, self.targets_p *
                                      self.pos_equal_one_for_reg_p, sigma) / self.pos_equal_one_sum_p
            self.reg_loss_p = tf.compat.v1.reduce_sum(self.reg_loss_p)
            
            #cyclist

            #Probability score map, scale = [None, 200/100, 176/120, 2]
            p_map_c = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv,
                           training=self.training, activation=False, bn=False, name='convc20')  #make output channel 4
            # Regression(residual) map, scale = [None, 200/100, 176/120, 14]
            r_map_c = ConvMD(2, 768, 14, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='convc21')  #make output channel 28
            #softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1]
            self.p_pos_c = tf.compat.v1.sigmoid(p_map_c,name='probc') # 
            #self.p_pos = tf.compat.v1.nn.softmax(p_map, dim=3)

            #stacking the pedestrain and cyclist regression and prob map together in axis 0
            r_map = tf.compat.v1.stack([r_map_p,r_map_c],axis=0,name='conv22')
            p_pos = tf.compat.v1.stack([self.p_pos_p,self.p_pos_c],axis=0,name='prob')
            
            self.output_shape = [cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH]
            #cross entropy 
            self.cls_pos_loss_c = (-self.pos_equal_one_c * tf.compat.v1.log(self.p_pos_c + small_addon_for_BCE)) / self.pos_equal_one_sum_c
        
            self.cls_neg_loss_c = (-self.neg_equal_one_c * tf.compat.v1.log(1 - self.p_pos_c + small_addon_for_BCE)) / self.neg_equal_one_sum_c

           
            self.cls_pos_loss_rec_c = tf.compat.v1.reduce_sum( self.cls_pos_loss_c)
            self.cls_neg_loss_rec_c = tf.compat.v1.reduce_sum( self.cls_neg_loss_c)

          
            self.reg_loss_c = smooth_l1(r_map_c * self.pos_equal_one_for_reg_c, self.targets_c *
                                      self.pos_equal_one_for_reg_c, sigma) / self.pos_equal_one_sum_c
            self.reg_loss_c = tf.compat.v1.reduce_sum(self.reg_loss_c)

            self.cls_loss_p = tf.compat.v1.reduce_sum( 1.5 * self.cls_pos_loss_p + 1.0 * self.cls_neg_loss_p )  #hyperparameters  alpha and beta
            self.cls_loss_c = tf.compat.v1.reduce_sum( 1.5 * self.cls_pos_loss_c + 1.0 * self.cls_neg_loss_c ) #hyperparameters   alpha1 and beta1
            
            self.cls_loss = tf.compat.v1.reduce_sum(1* self.cls_loss_p + 1.3 * self.cls_loss_c)  #hyperparameters A and B
            self.reg_loss = tf.compat.v1.reduce_sum(1 * self.reg_loss_p + 1.3 * self.reg_loss_c) #hyperparameters  A1 and B1

            self.loss = tf.compat.v1.reduce_sum(self.cls_loss + self.reg_loss)

            self.delta_output_p = r_map_p
            self.prob_output_p = self.p_pos_p
            self.delta_output_c = r_map_c
            self.prob_output_c = self.p_pos_c


def smooth_l1(deltas, targets, sigma=3.0):  #hyperparamter
    sigma2 = sigma * sigma
    diffs = tf.compat.v1.subtract(deltas, targets)
    smooth_l1_signs = tf.compat.v1.cast(tf.compat.v1.less(tf.compat.v1.abs(diffs), 1.0 / sigma2), tf.compat.v1.float32)

    smooth_l1_option1 = tf.compat.v1.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.compat.v1.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.compat.v1.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.compat.v1.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def ConvMD(M, Cin, Cout, k, s, p, input, training=True, activation=True, bn=True, name='conv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    with tf.compat.v1.variable_scope(name) as scope:
        if(M == 2):
            paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
            pad = tf.compat.v1.pad(input, paddings, "CONSTANT")
            temp_conv = tf.compat.v1.layers.conv2d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.compat.v1.AUTO_REUSE, name=scope)
        if(M == 3):
            paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
            pad = tf.compat.v1.pad(input, paddings, "CONSTANT")
            temp_conv = tf.compat.v1.layers.conv3d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.compat.v1.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.compat.v1.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.compat.v1.AUTO_REUSE, name=scope)
        if activation:
            return tf.compat.v1.nn.relu(temp_conv)
        else:
            return temp_conv

def Deconv2D(Cin, Cout, k, s, p, input, training=True, bn=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.compat.v1.pad(input, paddings, "CONSTANT")
    with tf.compat.v1.variable_scope(name) as scope:
        temp_conv = tf.compat.v1.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", reuse=tf.compat.v1.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.compat.v1.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.compat.v1.AUTO_REUSE, name=scope)
        return tf.compat.v1.nn.relu(temp_conv)


if(__name__ == "__main__"):
    m = MiddleAndRPN(tf.compat.v1.placeholder(
        tf.compat.v1.float32, [None, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))
