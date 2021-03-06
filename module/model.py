#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Create Mann model for Meta-learning based on LSTM model as controller.
# @Time    : 6/28/2019 3:40 PM
# @Author  : Gaopeng.Bai
# @File    : model.py
# @User    : baigaopeng
# @Software: PyCharm
# Reference:https://github.com/Gaopeng-Bai/MANN_model.git

import tensorflow as tf


class NTMOneShotLearningModel:
    def __init__(self, args):

        self.x_data = tf.placeholder(dtype=tf.float32,
                                     shape=[args.batch_size, args.seq_length], name="x_squences")
        self.x_label = tf.placeholder(dtype=tf.float32,
                                      shape=[args.batch_size, args.output_dim], name="x_label")
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[args.batch_size, args.output_dim], name="y")

        if args.model == 'LSTM':
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            import ntm.ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    read_head_num=args.read_head_num,
                                    write_head_num=args.write_head_num,
                                    addressing_mode='content_and_location',
                                    output_dim=args.output_dim)
        elif args.model == 'MANN':
            import ntm.mann_cell as mann_cell
            cell = mann_cell.MANNCell(rnn_size=args.rnn_size, memory_size=args.memory_size,
                                      memory_vector_dim=args.memory_vector_dim,
                                      head_num=args.read_head_num, rnn_layers=args.rnn_num_layers)

        state = cell.zero_state(args.batch_size, tf.float32)
        self.state_list = [state]  # For debugging
        b = tf.concat([self.x_data[:, :], self.x_label[:, :]], axis=1)
        output, state = cell(b, state)
        with tf.variable_scope("o2o", reuse=tf.AUTO_REUSE):
            o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

        output = tf.nn.softmax(output)
        self.state_list.append(state)
        self.o = tf.squeeze(output, name="output")

        eps = 1e-8
        self.learning_loss = -tf.reduce_mean(  # cross entropy function
            tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1])
        )
        self.accuracy, self.acc_op = tf.metrics.accuracy(labels=tf.argmax(self.y, 1),
                                                         predictions=tf.argmax(self.o, 1), name="accuracy")
        self.recall, self.rec_op = tf.metrics.recall_at_k(labels=tf.cast(self.y, tf.int64),
                                                          predictions=self.o, k=100)
        self.precision, self.pre_op = tf.metrics.precision(labels=tf.argmax(self.y, 1),
                                                           predictions=tf.argmax(self.o, 1), name="precision")

        tf.summary.scalar('learning_loss', self.learning_loss)
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Recall_k', self.recall)
        tf.summary.scalar('precision', self.precision)
        self.merged_summary_op = tf.summary.merge_all()

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(
            #     learning_rate=args.learning_rate, momentum=0.9, decay=0.95
            # )
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss, name="train_op")
