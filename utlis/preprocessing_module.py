#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Create batches from numpy files fit in machine model.
# @Time    : 6/28/2019 3:40 PM
# @Author  : Gaopeng.Bai
# @File    : preprocessing_module.py
# @User    : baigaopeng
# @Software: PyCharm
# Reference:**********************************************

import os
from typing import Any, Union, Iterable

import numpy as np
import math

from numpy.core.multiarray import ndarray

from dataLoader import dataLoader


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim,), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res.tolist()


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def reconver_list(seq, list):
    new = []
    recover = math.ceil(((seq - len(list)) / len(list)))
    for i in range(recover + 1):
        new.append(list)

    return np.array(new).reshape([-1])[:seq]


class preprocessing:
    # noinspection PyCompatibility
    x_array: ndarray
    # noinspection PyCompatibility
    test_data: Union[Union[ndarray, Iterable, int, float, tuple, dict, None], Any]
    # noinspection PyCompatibility
    playlist_number: int
    # noinspection PyCompatibility
    data: Union[Union[ndarray, Iterable, int, float, tuple, dict, None], Any]
    # noinspection PyCompatibility
    playlist_number_test: int
    # noinspection PyCompatibility
    file_number: int

    def __init__(self, tasks_size=16, batch_size=8, length=10810, test=True,
                 numpy_dir="../../data_resources/save_data/Tensor_numpy", seq_length=50):
        self.tasks_size = tasks_size
        self.files = []
        self.batch_size = batch_size

        self.seq = seq_length
        self.length = length
        self.test = test
        for dir, subdir, filename in os.walk(numpy_dir):
            if filename:
                for file in filename:
                    self.files.append(numpy_dir + '/' + file)

    def init_preprocessing(self):
        self.file_number = 0

        self.data = np.load(self.files[self.file_number], allow_pickle=True)
        self.playlist_number = len(self.data)
        if len(self.files) >= 1:
            self.test_data = np.load(self.files[(len(self.files) - 1)], allow_pickle=True)
            self.playlist_number_test = len(self.test_data)

    def temp_seq_length(self):
        a = []
        for pl in self.data:
            a.append(len(pl))

        a.sort()

    def normalization(self, playlist):
        """
        convert each value to 0-1 number. normalization algorithm: Linear transformation.
         y=(x-min)/(max-min).
        Args:
            playlist: input a playlist.
        Returns: normalized value list.
        """
        new_list = []
        for i in playlist:
            new_list.append(i / self.length)
        return new_list

    def create_batch(self, test_data=False):
        """
        Create a batch from batch table with batch size and tasks size represent a playlist.
        a batch(1-dim) represent a playlist rebuild no.(task size(2-dim)) of item.
        each item extend the length of seq length(3-dim).
        add second data that shift y label one bit to left.
        Args:
            test_data: fetch next test batch when True,
                       fetch next train batch when False.
        Returns: x_out, y_out as a batch with 2-dim(task size).
        """
        y_out = []
        x_out = []
        if test_data:
            for i in range(self.batch_size):
                x_test, y_test = self.fetch_batch(test_data)
                while len(x_test) < self.tasks_size:
                    x_test, y_test = self.fetch_batch(test_data)

                seq = np.random.randint(0, len(y_test), self.tasks_size)

                # random choice tasks size item.
                y_out.append([y_test[i] for i in seq])
                x_out.append([x_test[i] for i in seq])
                x_labels = y_out[-1:] + y_out[:-1]
        else:
            for i in range(self.batch_size):
                x, y = self.fetch_batch(test_data)
                while len(x) < self.tasks_size:
                    x, y = self.fetch_batch(test_data)

                seq = np.random.randint(0, len(y), self.tasks_size)

                # random choice tasks size item.
                y_out.append([y[i] for i in seq])
                x_out.append([x[i] for i in seq])

                x_labels = y_out[-1:]+y_out[:-1]

        return x_out, x_labels, y_out

    def next_batch(self, test_data=False):
        """
        Fetch next batch for test and train datasets
        Args:
            test_data: fetch next test batch when True,
                       fetch next train batch when False.
        Returns: a test or training data batch.
        """
        if test_data:
            if self.playlist_number_test >= self.batch_size:
                x, x_label, y = self.assign_to_final(test_data)
                return x, x_label, y
            else:
                self.read_file(test_data)
                x, x_label, y = self.assign_to_final(test_data)
                return x, x_label, y
        else:
            if self.playlist_number >= self.batch_size:
                x, x_label, y = self.assign_to_final(test_data)
                return x, x_label, y
            else:
                # read all files in one epoch.
                if self.read_file(test_data) is None:
                    return None, None, None
                else:
                    x, x_label, y = self.assign_to_final(test_data)
                    return x, x_label, y

    def assign_to_final(self, test_data):
        """
        Args:
            test_data: fetch next test batch when True,
                       fetch next train batch when False.
        Returns: a batch datasets.
        """
        x, y_shifted, y = self.create_batch(test_data)
        return np.array(x), np.array(y_shifted), np.array(y)

    def fetch_batch(self, test_data=False):
        """
        Convert playlist to batch table.
        Args:
            test_data: fetch batch table from test datasets.
        Returns:
        """
        y_output = []
        x_array = np.arange(self.seq, dtype=float).reshape(1, self.seq)
        if test_data:
            self.playlist_number_test -= 1
            playlist = self.test_data[self.playlist_number_test]
        else:
            self.playlist_number -= 1
            playlist = self.data[self.playlist_number]

        # remove None in playlists
        if self.test:
            while None in playlist:
                playlist.remove(None)

        playlist = self.normalization(playlist)

        for i in range(len(playlist) - 2):
            x = []
            y = playlist[i + 1]
            temp_list = playlist[:i + 1]

            if len(temp_list) >= self.seq:
                x.append(temp_list[:self.seq])
                x_array = np.insert(x_array, len(x_array), np.array(x[0]), axis=0)
            else:
                x.append(temp_list)
                while len(x[0]) < self.seq:
                    x[0].append(-1)
                x_array = np.insert(x_array, len(x_array), np.array(x[0]), axis=0)

            y_output.append(one_hot_encode(int(y * self.length), self.length))

        return x_array[1:, :].tolist(), y_output

    def read_file(self, test_data=False):
        """
        test dataset: refresh the data memory when finished a loop. Assigned the last one file as test dataset.
        train dataset: load training data memory from the next training data files.
        Args:
            test_data: refresh test file when True,
                       load next training file when False.
        Returns: when all training files been done return None
        """
        if test_data:
            self.test_data = np.load(self.files[(len(self.files) - 1)], allow_pickle=True)
            self.playlist_number_test = len(self.data)
        else:
            self.file_number += 1

            if self.file_number > len(self.files) - 2:
                return None
            else:
                self.data = np.load(self.files[self.file_number], allow_pickle=True)
                self.playlist_number = len(self.data)


if __name__ == '__main__':

    loader = preprocessing()
    loader.init_preprocessing()
    x, x_label, y = loader.next_batch(test_data=False)
    while x is not None:
        print(0)
        x, x_label, y = loader.next_batch(test_data=False)
    print("all files done")
