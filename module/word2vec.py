# -*- coding:utf-8 -*-
import numpy as np
import math
import os
import tensorflow as tf
import collections
import pickle as pkl


class word2vec():
    # 1、初始化参数
    def __init__(self,
                 vocab_list=None,  # 词典默认为None，需用户传入
                 embedding_size=200,  # 词向量维度，默认为200
                 win_len=3,  # 单边窗口长，也就是上下文，考虑某个词的前面3个词，后面3个词
                 num_sampled=1000,  # 负采样个数，默认1000
                 learning_rate=0.1,  # 学习率
                 logdir='summary/simple_word2vec',  # 模型保存目录
                 model_path="model/word2vec"  # 模型路径，用于训练好后加载模型用
                 ):

        # 获得模型的基本参数
        self.batch_size = None  # 一批中数据个数
        if os.listdir(model_path):  # 如果模型已经训练好，可以直接加载模型
            self.load_model(model_path)
        else:
            # model parameters
            assert type(vocab_list) == list
            # 如果vocab_list类型不是list，即：type(vocab_list) == list为假，我们直接让程序出发异常
            # 否则就可以继续运行
            self.vocab_list = vocab_list  # 词典
            self.vocab_size = vocab_list.__len__()  # 词典中词的总数
            self.embedding_size = embedding_size  # 词向量维度
            self.win_len = win_len  # 窗口大小
            self.num_sampled = num_sampled  # 负采样个数
            self.learning_rate = learning_rate  # 学习率
            self.logdir = logdir  # 目录
            self.save_model_path= model_path

            self.word2id = {}  # word => id 的映射
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]] = i

            self.train_words_num = 0  # 训练的单词对数
            self.train_sents_num = 0  # 训练的句子数
            self.train_times_num = 0  # 训练的次数（一次可以有多个句子）

            self.train_loss_records = collections.deque(maxlen=10)  # 保存最近10次的误差
            self.train_loss_k10 = 0

        self.build_graph()  # 构建图
        self.init_op()  # 操作初始化
        if os.listdir(model_path):  # 模型路径不为None
            tf_model_path = os.path.join(model_path, 'tf_vars')  # 路径拼接
            self.saver.restore(self.sess, tf_model_path)  # 恢复以前保存的变量

    # 2、图的定义
    def build_graph(self):
        self.graph = tf.Graph()  # 声明图，所有的变量和操作都定义在图上,使用新构建的图
        with self.graph.as_default():  # 定义graph图，在“graph”中定义运算和张量。
            # 此时在这个代码块中，使用的就是新的定义的图graph(相当于把默认图换成了graph)
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])  # 定义占位符，输入数据
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])  # 中心词
            # 定义词向量变量，形状为[vocab_size, embedding_size]，并用正态分布均值为-1.0，方差为1.0的随机数赋值
            # 其形状为[[],[],...,[]],行为每个词，列为每个词我们自定义的词向量维度
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)
            )
            # 定义权值变量，形状为[vocab_size, embedding_size]，并用正态分布均值为0.0，方差为stddev的随机数赋值
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
            # 定义偏置，形状为vocab_size行1列，初始化为0
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)  # batch_size

            # 得到NCE损失
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size
                )
            )

            # tensorboard 相关
            tf.summary.scalar('loss', self.loss)  # 让tensorflow记录参数

            # 根据 nce loss 来更新梯度和embedding，梯度下降法
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32, shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True)
                # square：求平方
                # reduce_sum：按行求和
                # sqrt：求根号
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)  # 求均值
            tf.summary.scalar('avg_vec_model', avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model  # 对embedding_dict向量正则化
            self.test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)  # 嵌入式张量列表中查找
            self.similarity = tf.matmul(self.test_embed, self.normed_embedding, transpose_b=True)
            # matmul：矩阵相乘
            # 变量初始化
            self.init = tf.global_variables_initializer()
            # 合并默认图中收集的所有摘要
            self.merged_summary_op = tf.summary.merge_all()
            # 保存模型
            self.saver = tf.train.Saver()

    # 3、操作初始化
    def init_op(self):
        # 创建会话
        self.sess = tf.Session(graph=self.graph)
        # 执行会话
        self.sess.run(self.init)
        # FileWriter类提供了一个机制来创建指定目录的事件文件，并添加摘要和事件给它
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    # 4、训练模型
    def train_by_sentence(self, input_sentence=[] , epoch=0):
        #  input_sentence: [sub_sent1, sub_sent2, ...]
        # 每个sub_sent是一个单词序列，例如['这次','大选','让']
        # sent_num = input_sentence.__len__()
        batch_inputs = []  # 输入数据，上下文
        batch_labels = []  # 中心词
        for sent in input_sentence:  # 遍历每个sub_sent
            for i in range(sent.__len__()):  # 遍历sub_sent中每个词
                # 窗口
                start = max(0, i - self.win_len)  # 开始下标 窗口为 [-win_len,+win_len],总计长2*win_len+1
                end = min(sent.__len__(), i + self.win_len + 1)  # 结束下标
                for index in range(start, end):  # 遍历窗口内的词
                    if index == i:  # 跳过中心词
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])  # 上下文
                        label_id = self.word2id.get(sent[index])  # 中心词
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)  # 上下文id添加到batch_inputs
                        batch_labels.append(label_id)  # 中心词id添加到batch_labels
        if len(batch_inputs) == 0:  # 直到batch_inputs为空，结束函数
            return
        batch_inputs = np.array(batch_inputs, dtype=np.int32)  # 转为数组
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])  # 改变形状
        # 模型需传入的输入数据
        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        # 开启会话，并传入数据
        _, loss_val, summary_str = self.sess.run([self.train_op, self.loss, self.merged_summary_op],
                                                 feed_dict=feed_dict)

        # 训练损失
        self.train_loss_records.append(loss_val)
        # self.train_loss_k10 = sum(self.train_loss_records)/self.train_loss_records.__len__()
        self.train_loss_k10 = np.mean(self.train_loss_records)
        if self.train_sents_num % 1000 == 0:
            #  add_summary：FileWriter类中的函数，将摘要协议缓冲区添加到事件文件中
            self.summary_writer.add_summary(summary_str, self.train_sents_num)
            print("{a} sentences dealed, loss: {b} , Epoch:{num}"
                  .format(a=self.train_sents_num, b=self.train_loss_k10, num=epoch))

        # 训练次数
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    # 5、计算相似概率，测试词的id，只选取相似概率前10个
    def cal_similarity(self, test_word_id_list, top_k=10):
        # 相似度矩阵
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id: test_word_id_list})
        sim_mean = np.mean(sim_matrix)  # 按行，求均值
        sim_var = np.mean(np.square(sim_matrix - sim_mean))  # 按行求方差
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])  # 测试单词
            nearst_id = (-sim_matrix[i, :]).argsort()[1:top_k + 1]  # 最相似的10个词的id，并排序
            nearst_word = [self.vocab_list[x] for x in nearst_id]  # 最相似的词
            near_words.append(nearst_word)  # 添加到相似词near_words列表中
        return test_words, near_words, sim_mean, sim_var  # 返回测试词，相似词，相似度均值，相似度方差

    # 6、模型保存
    def save_model(self):

        if os.path.isfile(self.save_model_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)

        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',  # int       model parameters
                     'vocab_list',  # list
                     'learning_rate',  # int
                     'word2id',  # dict
                     'embedding_size',  # int
                     'logdir',  # str
                     'win_len',  # int
                     'num_sampled',  # int
                     'train_words_num',  # int       train info
                     'train_sents_num',  # int
                     'train_times_num',  # int
                     'train_loss_records',  # int   train loss
                     'train_loss_k10',  # int
                     ]
        for var in var_names:
            model[var] = eval('self.' + var)

        param_path = os.path.join(self.save_model_path, 'params.pkl')  # 路径拼接
        if os.path.exists(param_path):  # 如果路径存在就删除
            os.remove(param_path)
        with open(param_path, 'wb') as f:  # 否则以'wb'方式打开文件
            pkl.dump(model, f)  # 保存模型

        # 记录tf模型
        tf_path = os.path.join(self.save_model_path, 'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess, tf_path)

    # 7、模型加载
    def load_model(self, model_path):
        if not os.path.exists(model_path):  # 如果不存在，即没有训练好的模型
            raise RuntimeError('file not exists')

        # 否则拼接路径
        param_path = os.path.join(model_path, 'params.pkl')
        with open(param_path, 'rb') as f:  # 以'rb'方式打开文件，读取参数
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']