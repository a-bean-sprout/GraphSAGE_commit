# -*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer): # A(B)代表继承关系
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):  # 类实例化将直接调用此方法
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))  # 随机打乱邻居节点
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])  # 取出[0,0]到[-1,num_sample]  随机获取邻居节点
        return adj_lists
