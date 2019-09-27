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
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def reconver_list(seq, list):
    new = []
    recover = math.ceil(((seq - len(list)) / len(list)))
    for i in range(recover + 1):
        new.append(list)

    return np.array(new).reshape([-1])[:seq]


class preprocessing:
    test_data: Union[Union[ndarray, Iterable, int, float, tuple, dict, None], Any]
    playlist_number: int
    data: Union[Union[ndarray, Iterable, int, float, tuple, dict, None], Any]
    playlist_number_test: int
    file_number: int

    def __init__(self, tasks_size=16, batch_size=2, length=10810, test=True, random_number_files=19, file_number=20,
                 numpy_dir="../save_data/Tensor_numpy", seq_length=50):
        self.tasks_size = tasks_size
        self.files = []
        self.x = []
        self.batch_size = batch_size
        self.temp_list = []
        self.y_output = []
        self.x_label = np.random.randint(0, length, [batch_size, tasks_size, length])

        self.seq = seq_length
        self.length = length
        self.test = test
        for dir, subdir, filename in os.walk(numpy_dir):
            if filename:
                for file in filename:
                    self.files.append(numpy_dir + '/' + file)
            else:
                loader = dataLoader(random_number_files=random_number_files, file_number=file_number)
                self.length = loader.vocabulary_size

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

    def standardization(self, playlist):
        new_list = []
        for i in playlist:
            new_list.append(i / self.length)
        return new_list

    def create_batch(self, test_data=False):
        y_out = []
        x_out = []

        if test_data:
            for i in range(self.batch_size):
                x_test, y_test = self.fetch_batch(test_data=True)
                while len(x_test) < self.tasks_size:
                    x_test, y_test = self.fetch_batch(test_data=True)

                seq = np.random.randint(0, len(y_test), self.tasks_size)

                # random choice tasks size item.
                y_out.append([y_test[i] for i in seq])
                x_out.append([x_test[i] for i in seq])
        else:
            for i in range(self.batch_size):
                x, y = self.fetch_batch()
                while len(x) < self.tasks_size:
                    x, y = self.fetch_batch()

                seq = np.random.randint(0, len(y), self.tasks_size)

                # random choice tasks size item.
                y_out.append([y[i] for i in seq])
                x_out.append([x[i] for i in seq])

        return x_out, y_out

    # fetch a batch.
    # @ test_data: True. fetch a test data batch.
    def next_batch(self, test_data=False):

        if test_data:
            if self.playlist_number_test >= self.batch_size:
                x, y = self.create_batch(test_data)
            else:
                self.read_file(test_data)
                x, y = self.create_batch(test_data)

            return np.array(x), np.array(y)
        else:
            if self.playlist_number >= self.batch_size:
                x, y = self.create_batch(test_data)
                return np.array(x), np.array(y)
            else:
                # read all files in one epoch.
                if self.read_file(test_data) is None:
                    return None, None
                else:
                    x, y = self.create_batch(test_data)
                    return np.array(x), np.array(y)

    def fetch_batch(self, test_data=False):
        self.y_output.clear()
        self.temp_list.clear()
        self.x_array = np.arange(self.seq, dtype=float).reshape(1, self.seq)

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

        playlist = self.standardization(playlist)

        for i in range(len(playlist) - 2):
            self.x.clear()
            y = playlist[i + 1]
            self.temp_list = playlist[:i + 1]

            if len(self.temp_list) >= self.seq:
                self.x.append(self.temp_list[:self.seq])
                self.x_array = np.insert(self.x_array, len(self.x_array), np.array(self.x[0]), axis=0)
            else:
                self.x.append(self.temp_list)
                while len(self.x[0]) < self.seq:
                    self.x[0].append(-1)
                self.x_array = np.insert(self.x_array, len(self.x_array), np.array(self.x[0]), axis=0)

            self.y_output.append(one_hot_encode(int(y * self.length), self.length))

            y_output = np.array(self.y_output)

        return self.x_array[1:, :], y_output

    def read_file(self, test_data=False):
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
    x, y = loader.next_batch(test_data=False)
    while x is not None:
        print(0)
        x, y = loader.next_batch(test_data=False)
    print("all files done")
