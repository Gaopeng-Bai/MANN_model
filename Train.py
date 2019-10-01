#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:https://github.com/snowkylin/ntm.git
# @Time    : 6/28/2019 3:40 PM
# @Author  : Gaopeng.Bai
# @File    : Train.py
# @User    : baigaopeng
# @Software: PyCharm
# Reference:**********************************************

import argparse
import os
from module.model import NTMOneShotLearningModel
from utlis.preprocessing_module import preprocessing as pre
import tensorflow as tf
from tensorflow.python import debug as tf_debug


def main():
    # deactivate the warnings for "teh tf library wasn't co to use SSE instructions"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")

    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN_NEW or NTM')

    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--training', default=True)

    parser.add_argument('--save_dir', default="../data_resources/save_data")
    parser.add_argument('--model_dir', default="../data_resources/summary/model")
    parser.add_argument('--numpy_dir', default="../data_resources/save_data/Tensor_numpy")
    parser.add_argument('--tensorboard_dir', default='../data_resources/summary')

    parser.add_argument('--number_files', default=10, help="For dataLoader, the number of files read once")

    parser.add_argument('--output_dim', default=10810)
    parser.add_argument('--seq_length', default=50)
    parser.add_argument('--tasks_size', default=10)

    parser.add_argument('--num_epoches', default=80000)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--learning_rate', default=5e-4)
    parser.add_argument('--rnn_size', default=128)
    parser.add_argument('--rnn_num_layers', default=1)
    parser.add_argument('--output_keep_prob', default=1.0,
                        help='probability of keeping weights in the hidden layer')

    parser.add_argument('--memory_size', default=256)
    parser.add_argument('--read_head_num', default=4)
    parser.add_argument('--memory_vector_dim', default=450)

    parser.add_argument('--shift_range', default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')

    parser.add_argument('--test_batch_num', default=100)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        predict(args)


def train(args):
    model = NTMOneShotLearningModel(args)

    data_loader = pre(batch_size=args.batch_size, length=args.output_dim, tasks_size=args.tasks_size, numpy_dir=args.numpy_dir,
                      seq_length=args.seq_length)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        if args.debug:
            sess.run(init)
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if args.restore_training:
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(args.save_dir + '/' + args.model))
        else:
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables(), tf.local_variables())
        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model + '/' + "train", sess.graph)
        test_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model + '/' + "test", sess.graph)

        number_test = 0
        for e in range(args.num_epoches):
            data_loader.init_preprocessing()
            x, x_label, y = data_loader.next_batch(test_data=False)
            while x is not None:
                number_test += 1
                # test
                if number_test % 10 == 0:
                    x, x_label, y = data_loader.next_batch(test_data=True)
                    feed_dict = {model.x_data: x, model.x_label: x_label, model.y: y}
                    learning_loss, accuracy, recall, precision = sess.run(
                        [model.learning_loss, model.accuracy, model.recall, model.precision], feed_dict=feed_dict)

                    acc_op, recall_op, pre_op, merged_summary = sess.run(
                        [model.acc_op, model.rec_op, model.pre_op, model.merged_summary_op], feed_dict=feed_dict)

                    test_writer.add_summary(merged_summary, number_test)

                    print(
                        "Epochs {}/{}, Files {}/{}, learning_loss:{}, Accuracy :{}, Recall:{}, "
                        "precision:{} "
                            .format(e, args.num_epoches, len(data_loader.files), number_test, learning_loss,
                                    '%.2f%%' % (accuracy * 100), recall, precision))
                    # print("acc_op:{}, recall_op :{}".format(acc_op,recall_op))
                else:
                    # Train
                    feed_dict = {model.x_data: x, model.x_label: x_label, model.y: y}
                    _, merged = sess.run([model.train_op, model.merged_summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(merged, number_test)

                x, x_label, y = data_loader.next_batch(test_data=False)

            # save model
            if e % 100 == 0:
                print("model saver :{} times".format(e/100))
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel')


def predict(args, x):
    # init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        meta = [fn for fn in os.listdir(args.save_dir + '/' + args.model) if fn.endswith('meta')]
        saver = tf.train.import_meta_graph(args.save_dir + '/' + args.model + meta[0])
        saver.restore(sess, tf.train.latest_checkpoint(args.save_dir + '/' + args.model))
        prediction = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('x_squences').outputs[0]

        sp_predict = sess.run(prediction, feed_dict={input_x: x})


if __name__ == '__main__':
    main()
