#!/usr/bin/env python
# -*- coding:UTF-8 -*-

#import mayavi.mlab as mlab
import sys
import os
import tensorflow as tf
import cv2
import numpy as np
#import torch

from numba import jit

import pyximport
pyximport.install()

from config import cfg
from utils import *
from model.group_pointcloud import FeatureNet
from model.rpn import MiddleAndRPN

#from visual_utils import visualize_utils as V


class RPN3D(object):

    def __init__(self,
                 cls='Car',
                 single_batch_size=2,  # batch_size_per_gpu
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 alpha=1.5,
                 beta=1,
                 avail_gpus=['0']):
        # hyper parameters and status
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.compat.v1.Variable(
            float(learning_rate), trainable=False, dtype=tf.compat.v1.float32)
        self.global_step = tf.compat.v1.Variable(1, trainable=False)
        self.epoch = tf.compat.v1.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.beta = beta
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [ self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01  ]  #changed the learning rate 
        lr = tf.compat.v1.train.piecewise_constant(self.epoch, boundaries, values)

        # build graph
        # input placeholders
        self.is_train = tf.compat.v1.placeholder(tf.compat.v1.bool, name='phase')

        self.vox_feature = []
        self.vox_number = []
        self.vox_coordinate = []
        #pedestrian
        self.targets = [[] for i in range(2)]
        self.pos_equal_one = [[] for i in range(2)]
        self.pos_equal_one_sum = [[] for i in range(2)]
        self.pos_equal_one_for_reg = [[] for i in range(2)]
        self.neg_equal_one = [[] for i in range(2)]
        self.neg_equal_one_sum = [[] for i in range(2)]

        self.anchors = [[] for i in range(2)]
        self.delta_output = [[] for i in range(2)]
        self.prob_output = [[] for i in range(2)]
        
        self.opt = tf.compat.v1.train.AdamOptimizer(lr)
        self.gradient_norm = []
        self.tower_grads = []
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.compat.v1.device('/gpu:{}'.format(dev)), tf.compat.v1.name_scope('gpu_{}'.format(dev)):
                    #must use name scope here since we do not want to create new variables
                    #graph
                    feature = FeatureNet(
                        training=self.is_train, batch_size=self.single_batch_size)
                    rpn = MiddleAndRPN(
                        input=feature.outputs, alpha=self.alpha, beta=self.beta, training=self.is_train)
                    tf.compat.v1.get_variable_scope().reuse_variables()
                    #input
                    self.vox_feature.append(feature.feature)
                    self.vox_number.append(feature.number)
                    self.vox_coordinate.append(feature.coordinate)
                    #pedestrian
                    self.targets[0].append(rpn.targets_p)
                    self.pos_equal_one[0].append(rpn.pos_equal_one_p)
                    self.pos_equal_one_sum[0].append(rpn.pos_equal_one_sum_p)
                    self.pos_equal_one_for_reg[0].append(
                        rpn.pos_equal_one_for_reg_p)
                    self.neg_equal_one[0].append(rpn.neg_equal_one_p)
                    self.neg_equal_one_sum[0].append(rpn.neg_equal_one_sum_p)
                    #cyclist
                    self.targets[1].append(rpn.targets_c)
                    self.pos_equal_one[1].append(rpn.pos_equal_one_c)
                    self.pos_equal_one_sum[1].append(rpn.pos_equal_one_sum_c)
                    self.pos_equal_one_for_reg[1].append(
                        rpn.pos_equal_one_for_reg_c)
                    self.neg_equal_one[1].append(rpn.neg_equal_one_c)
                    self.neg_equal_one_sum[1].append(rpn.neg_equal_one_sum_c)
                    # output
                    feature_output = feature.outputs
                    delta_output_p = rpn.delta_output_p
                    prob_output_p = rpn.prob_output_p
                    delta_output_c = rpn.delta_output_c
                    prob_output_c = rpn.prob_output_c
                    # loss and grad
                    if idx == 0:
                        self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

                    self.loss = rpn.loss
                    self.reg_loss = rpn.reg_loss
                    self.cls_loss = rpn.cls_loss
                    self.cls_pos_loss_p = rpn.cls_pos_loss_rec_p
                    self.cls_neg_loss_p = rpn.cls_neg_loss_rec_p
                    self.cls_pos_loss_c = rpn.cls_pos_loss_rec_c
                    self.cls_neg_loss_c = rpn.cls_neg_loss_rec_c
                    self.reg_loss_p = rpn.reg_loss_p
                    self.reg_loss_c = rpn.reg_loss_c
                    self.params = tf.compat.v1.trainable_variables()
                    gradients = tf.compat.v1.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.compat.v1.clip_by_global_norm(
                        gradients, max_gradient_norm)

                    self.delta_output[0].append(delta_output_p)
                    self.prob_output[0].append(prob_output_p)
                    self.delta_output[1].append(delta_output_c)
                    self.prob_output[1].append(prob_output_c)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.rpn_output_shape = rpn.output_shape

        self.vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        # loss and optimizer
        # self.xxxloss is only the loss for the lowest tower
        with tf.compat.v1.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = average_gradients(self.tower_grads)
            self.update = [self.opt.apply_gradients(
                zip(self.grads, self.params), global_step=self.global_step)]
            self.gradient_norm = tf.compat.v1.group(*self.gradient_norm)

        self.update.extend(self.extra_update_ops)
        self.update = tf.compat.v1.group(*self.update)

        self.delta_output[0] = tf.compat.v1.concat(self.delta_output[0], axis=0)
        self.prob_output[0] = tf.compat.v1.concat(self.prob_output[0], axis=0)

        self.delta_output[1] = tf.compat.v1.concat(self.delta_output[1], axis=0)
        self.prob_output[1] = tf.compat.v1.concat(self.prob_output[1], axis=0)
  
        # concatenate both cyclist and pedestrian tensors here 
        # concatenate prob_map of cyclist and pedestrian and reg_map of cyclist and pedestrian 
        
        
        self.anchors[0] = cal_anchors(0.8) #pedestrian
        self.anchors[1] = cal_anchors(1.7) #cyclist
        # for predict and image summary
        self.rgb = tf.compat.v1.placeholder(
            tf.compat.v1.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
        self.bv = tf.compat.v1.placeholder(tf.compat.v1.uint8, [
                                 None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3])
        self.bv_heatmap = tf.compat.v1.placeholder(tf.compat.v1.uint8, [
            None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3])
        self.boxes2d = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 4],name="boxes2d")
        self.boxes2d_scores = tf.compat.v1.placeholder(tf.compat.v1.float32, [None],name="boxes2d_scores")

        # NMS(2D) write 3D NMS 
        with tf.compat.v1.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.box2d_ind_after_nms = tf.compat.v1.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
                
        with tf.compat.v1.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.box2d_ind_after_nms = tf.compat.v1.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)

        # summary and saver
        self.saver = tf.compat.v1.train.Saver(write_version=tf.compat.v1.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar('train/loss', self.loss),
            tf.compat.v1.summary.scalar('train/reg_loss', self.reg_loss),
            tf.compat.v1.summary.scalar('train/cls_loss', self.cls_loss),
            tf.compat.v1.summary.scalar('train/cls_pos_loss_p', self.cls_pos_loss_p),
            tf.compat.v1.summary.scalar('train/cls_neg_loss_p', self.cls_neg_loss_p),
            tf.compat.v1.summary.scalar('train/cls_pos_loss_c', self.cls_pos_loss_c),
            tf.compat.v1.summary.scalar('train/cls_neg_loss_c', self.cls_neg_loss_c),
            tf.compat.v1.summary.scalar('train/reg_loss_p', self.reg_loss_p),
            tf.compat.v1.summary.scalar('train/reg_loss_c', self.reg_loss_c),
            *[tf.compat.v1.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

        self.validate_summary = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar('validate/loss', self.loss),
            tf.compat.v1.summary.scalar('validate/reg_loss', self.reg_loss),
            tf.compat.v1.summary.scalar('validate/cls_loss', self.cls_loss),
            tf.compat.v1.summary.scalar('validate/cls_pos_loss_p', self.cls_pos_loss_p),
            tf.compat.v1.summary.scalar('validate/cls_neg_loss_p', self.cls_neg_loss_p),
            tf.compat.v1.summary.scalar('validate/cls_pos_loss_c', self.cls_pos_loss_c),
            tf.compat.v1.summary.scalar('validate/cls_neg_loss_c', self.cls_neg_loss_c),
            tf.compat.v1.summary.scalar('train/reg_loss_p', self.reg_loss_p),
            tf.compat.v1.summary.scalar('train/reg_loss_c', self.reg_loss_c)
        ])

        # TODO: bird_view_summary and front_view_summary

        self.predict_summary = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.image('predict/bird_view_lidar', self.bv),
            tf.compat.v1.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
            tf.compat.v1.summary.image('predict/front_view_rgb', self.rgb),
        ])

    def train_step(self, session, data, train=False, summary=False):
        # input:
        #     (N) tag
        #     (N, N') label
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        tag = data[0]
        label = data[1] ##load pedestrian and cyclist label at the time of loading make tuple of list 
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        print('train', tag)
        #pedestrian
        pos_equal_one_p, neg_equal_one_p, targets_p = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors[0], cls='Pedestrian', coordinate='lidar') #change the detect_object
        pos_equal_one_for_reg_p = np.concatenate(
            [np.tile(pos_equal_one_p[..., [0]], 7), np.tile(pos_equal_one_p[..., [1]], 7)], axis=-1)
        pos_equal_one_sum_p = np.clip(np.sum(pos_equal_one_p, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum_p = np.clip(np.sum(neg_equal_one_p, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        #cyclist
        pos_equal_one_c, neg_equal_one_c, targets_c = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors[1], cls='Cyclist', coordinate='lidar')
        pos_equal_one_for_reg_c = np.concatenate(
            [np.tile(pos_equal_one_c[..., [0]], 7), np.tile(pos_equal_one_c[..., [1]], 7)], axis=-1)
        pos_equal_one_sum_c = np.clip(np.sum(pos_equal_one_c, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum_c = np.clip(np.sum(neg_equal_one_c, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)


        input_feed = {}
        input_feed[self.is_train] = True
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            #Pedestrian
            input_feed[self.targets[0][idx]] = targets_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[0][idx]] = pos_equal_one_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[0][idx]] = pos_equal_one_sum_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[0][idx]] = pos_equal_one_for_reg_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[0][idx]] = neg_equal_one_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[0][idx]] = neg_equal_one_sum_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            #Cyclist
            input_feed[self.targets[1][idx]] = targets_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[1][idx]] = pos_equal_one_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[1][idx]] = pos_equal_one_sum_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[1][idx]] = pos_equal_one_for_reg_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[1][idx]] = neg_equal_one_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[1][idx]] = neg_equal_one_sum_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            
        if train:
            output_feed = [self.loss, self.reg_loss,
                           self.cls_loss, self.cls_pos_loss_p, self.cls_neg_loss_p,self.cls_pos_loss_c,self.cls_neg_loss_c,self.reg_loss_p,self.reg_loss_c,  self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.reg_loss, self.cls_loss, self.cls_pos_loss_p,self.cls_neg_loss_p,self.cls_pos_loss_c,self.cls_neg_loss_c, self.reg_loss_p,self.reg_loss_c]
        if summary:
            output_feed.append(self.train_summary)
        # TODO: multi-gpu support for test and predict step
        return session.run(output_feed, input_feed)

    def validate_step(self, session, data, summary=False):
        # input:
        #     (N) tag
        #     (N, N') label
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        print('valid', tag)
        #Pedestrian
        pos_equal_one_p, neg_equal_one_p, targets_p = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors[0])
        pos_equal_one_for_reg_p = np.concatenate(
            [np.tile(pos_equal_one_p[..., [0]], 7), np.tile(pos_equal_one_p[..., [1]], 7)], axis=-1)
        pos_equal_one_sum_p = np.clip(np.sum(pos_equal_one_p, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum_p = np.clip(np.sum(neg_equal_one_p, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        #Cyclist
        pos_equal_one_c, neg_equal_one_c, targets_c = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors[1])
        pos_equal_one_for_reg_c = np.concatenate(
            [np.tile(pos_equal_one_c[..., [0]], 7), np.tile(pos_equal_one_c[..., [1]], 7)], axis=-1)
        pos_equal_one_sum_c = np.clip(np.sum(pos_equal_one_c, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum_c = np.clip(np.sum(neg_equal_one_c, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            #Pedestrian
            input_feed[self.targets[0][idx]] = targets_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[0][idx]] = pos_equal_one_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[0][idx]] = pos_equal_one_sum_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[0][idx]] = pos_equal_one_for_reg_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[0][idx]] = neg_equal_one_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[0][idx]] = neg_equal_one_sum_p[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            #Cyclist
            input_feed[self.targets[1][idx]] = targets_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[1][idx]] = pos_equal_one_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[1][idx]] = pos_equal_one_sum_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[1][idx]] = pos_equal_one_for_reg_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[1][idx]] = neg_equal_one_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[1][idx]] = neg_equal_one_sum_c[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            

        output_feed = [self.loss, self.reg_loss, self.cls_loss,self.cls_pos_loss_p,self.cls_neg_loss_p,self.cls_pos_loss_c,self.cls_neg_loss_c,self.reg_loss_p,self.reg_loss_c]
        if summary:
            output_feed.append(self.validate_summary)
        return session.run(output_feed, input_feed)

    def predict_step(self, session, data, summary=False, vis=False):
        # input:
        #     (N) tag
        #     (N, N') label(can be empty)
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        #     img (N, w, l, 3)
        #     lidar (N, N', 4)
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional)
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]
        classes = ['Pedestrian','Cyclist']
        ret_box3d_score =  [[] for i in range(2)]
        front_images, bird_views, heatmaps =  [[] for i in range(2)], [[] for i in range(2)], [[] for i in range(2)]
        print('predict', tag)
        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]

        for j in range(len(classes)):
            self.cls = classes[j]
            print(self.cls)
            print(j)
            if summary or vis:
                batch_gt_boxes3d = label_to_gt_box3d(
                        label, cls=classes[j], coordinate='lidar')
            output_feed = [self.prob_output[j], self.delta_output[j]]
            probs, deltas = session.run(output_feed, input_feed)
          
            #BOTTLENECK
            batch_boxes3d = delta_to_boxes3d(
                    deltas, self.anchors[j], coordinate='lidar')
            
                
            batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
            batch_probs = probs.reshape(
                    (len(self.avail_gpus) * self.single_batch_size, -1))
            
            # NMS
            ret_box3d = []
            ret_score = []
            for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
                
                # remove box with low score
                ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
                tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
                tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
                tmp_scores = batch_probs[0, ind]   
            
                # TODO: if possible, use rotate NMS
                boxes2d = corner_to_standup_box2d(
                    center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
                ind = session.run(self.box2d_ind_after_nms, {
                    self.boxes2d: boxes2d,
                    self.boxes2d_scores: tmp_scores
                })
                tmp_boxes3d = tmp_boxes3d[ind, ...]
                tmp_scores = tmp_scores[ind]
                ret_box3d.append(tmp_boxes3d)
                ret_score.append(tmp_scores)

                #start from here, for testing 
                '''this code is only for visualization for model prediction'''
                # points_ = torch.from_numpy(lidar)
                # tmp_boxes3d = center_to_corner_box3d(tmp_boxes3d)  #to rotate the bounding boxes
                # pred_boxes = torch.from_numpy(tmp_boxes3d)
                # pred_scores = torch.from_numpy(tmp_scores)
                # lidar_ = lidar[0,:,:]
                # points_ = torch.from_numpy(lidar_)
                # print(pred_boxes.shape)
                # V.draw_scenes(points=points_,ref_boxes=pred_boxes,ref_scores=pred_scores)
                # mlab.show(stop=True)

                #sys.exit()  #use this when doing testing 

            
            for boxes3d, scores in zip(ret_box3d, ret_score):
                ret_box3d_score[j].append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                        boxes3d, scores[:, np.newaxis]], axis=-1))
            
            if summary:
                #only summary 1 in a batch
                cur_tag = tag[0]
                P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
                
                front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0],
                                                            batch_gt_boxes3d[0], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                    
                bird_view = lidar_to_bird_view_img(
                        lidar[0], factor=cfg.BV_LOG_FACTOR)
                        
                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0],
                                                            batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                    
                heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
                
                ret_summary[j] = session.run(self.predict_summary, {
                        self.rgb: front_image[np.newaxis, ...],
                        self.bv: bird_view[np.newaxis, ...],
                        self.bv_heatmap: heatmap[np.newaxis, ...]
                    })

                #return tag, ret_box3d_score, ret_summary
                
            if vis:
                for i in range(len(img)):
                    cur_tag = tag[i]
                    P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
                   
                    front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i],
                                                        batch_gt_boxes3d[i], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                                                        
                    bird_view = lidar_to_bird_view_img(
                                                        lidar[i], factor=cfg.BV_LOG_FACTOR)
                                                        
                    bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i],
                                                        batch_gt_boxes3d[i], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                        
                    heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)
                        
                    front_images[j].append(front_image)
                    bird_views[j].append(bird_view)
                    heatmaps[j].append(heatmap)
                print('vis ')
                #return tag, ret_box3d_score, front_images, bird_views, heatmaps
        if summary:
            return tag, ret_box3d_score, ret_summary
        if vis:
            return tag, ret_box3d_score, front_images, bird_views, heatmaps

        #print(len(ret_box3d_score)) # 2
        
        return tag, ret_box3d_score


def average_gradients(tower_grads):
    # ref:
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.compat.v1.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.compat.v1.concat(axis=0, values=grads)
        grad = tf.compat.v1.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    pass


