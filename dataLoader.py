import numpy as np
import json
import os
import time
from six.moves import cPickle


def get_File_size(filePath):
    filePath = np.unicode(filePath)
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)

    return round(fsize, 2)


# check dictionary whether duplicated
def check_duplicated_dict(id2vocab_file):
    with open(id2vocab_file, 'rb') as f:
        sVocab = cPickle.load(f)
    d2id = dict(zip(sVocab, range(len(sVocab))))
    return len(d2id) != len(set(d2id.values()))


# Load  .json data into numpy data by playlists track ids. Assign sequences number to indicated each songs.
# Store the mapping between numbers and songs in dictionary
class dataLoader:
    # @test: True. generate only 10 playlists.
    # @random_number_files: int. The number of .json files in one npy file.
    # @file_number: int.    The number of .json to be generated in total.
    # @ data_dir: the resources files in local data file.
    # @ save_dir: save dir in local save_data dir.
    def __init__(self, data_dir='../data_resources/data', save_dir='../data_resources/save_data',
                 test=True, random_number_files=1, file_number=2):
        self.save_vocab2id_dir = save_dir + '/' + 'vocab2id'
        self.save_id2vocab_dir = save_dir + '/' + 'id2word'
        self.save_tensor_dir = save_dir + '/' + 'Tensor_numpy'
        self.id2vocab_file = os.path.join(self.save_id2vocab_dir, "vocab.pkl")
        self.vocab2id_file = os.path.join(self.save_vocab2id_dir, "vocab.pkl")
        self.temp_file = os.path.join(save_dir, "temp.txt")

        self.random_number_files = random_number_files
        # find all files name and store in list.
        self.file_name = []
        # dictionary path
        self.dataArray = np.arange(0)

        self.numberOfiles = file_number
        self.test = test
        # all playlist as list stored
        self.data = []
        # all characters as array stored
        self.char_form = ''
        # store files for reading only once
        self.files = []

        self.playlist = -1
        # value array convert char to number store in list
        self.valueArray = []

        for dir, subdir, filename in os.walk(data_dir):
            for file in filename:
                self.file_name.append(open(file=dir + '/' + file, mode='r', errors='ignore'))

        # store dictionary to check the sequence of songs and tensor data to train module.
        self.read_file_to_store(self.file_name, self.random_number_files)

    def read_file_to_store(self, data, random_number_files=1):
        self.count_file = 0

        if not os.path.isdir(self.save_vocab2id_dir):
            os.makedirs(self.save_vocab2id_dir)
        if not os.path.isdir(self.save_tensor_dir):
            os.makedirs(self.save_tensor_dir)
        if not os.path.isdir(self.save_id2vocab_dir):
            os.makedirs(self.save_id2vocab_dir)
        if os.path.exists(self.temp_file):
            with open(self.temp_file, mode='r') as f:
                contents = f.readlines()
                for val in contents:
                    val.strip('\n')
                    self.files.append(int(val))

        while len(self.files) < self.numberOfiles:
            # clear buffer
            print(len(self.files))
            self.tensor = np.arange(0)
            self.valueArray.clear()
            # save dir setting

            current = time.time()

            self.tensor_file = os.path.join(self.save_tensor_dir, "data" + str(current) + ".npy")
            # random choice files
            # @n_classes the number of files
            for i in np.random.choice(range(len(data)), size=random_number_files, replace=False):
                if i not in self.files:
                    self.dataArray = self.read_json_file(data[i], i)
                    # self.playlist += 999
                if self.count_file >= random_number_files:
                    # print(self.dataValue)
                    self.count_file = 0
                    self.dictionary_update(self.dataArray, self.id2vocab_file, self.vocab2id_file, self.tensor_file)
                    self.dataArray = np.arange(0)
                    self.data.clear()
                    self.playlist = -1

                    with open(self.temp_file, mode='w') as f:
                        for val in self.files:
                            f.write(str(val))
                            f.write('\n')

    # read json files from self_filename that stored all files name
    def read_json_file(self, Filename, file):
        with Filename as f:
            try:
                dataStore = json.load(f, strict=False)
            except:
                print("Json decode error " + str(file))
            else:
                self.count_file += 1
                self.files.append(file)
                for i, name in enumerate(dataStore["playlists"]):
                    if self.test:
                        if i == 100:
                            break
                        else:
                            self.playlist += 1
                            self.data.append([])
                            for j, tracks in enumerate(dataStore["playlists"][i]["tracks"]):
                                self.data[self.playlist].append(
                                    dataStore["playlists"][i]["tracks"][j]["track_uri"][14:])
                    else:
                        self.playlist += 1
                        self.data.append([])
                        for j, tracks in enumerate(dataStore["playlists"][i]["tracks"]):
                            self.data[self.playlist].append(dataStore["playlists"][i]["tracks"][j]["track_uri"][14:])

                return np.array(self.data)

    # convert all data what already had into string to extract dictionary.
    def to_str(self, Array):
        # convert array to string for dictionary_update
        for L in Array:
            for value in L:
                self.char_form = self.char_form + ''.join(value)
            self.char_form = self.char_form + '\n'

        return self.char_form

    def check_all_file_size(self, dirR):
        for dir, subdir, filename in os.walk(dirR):
            if filename:
                for file in filename:
                    path = self.save_vocab2id_dir + '/' + file
                    size = get_File_size(path)
                    if size < 60:
                        return file
            else:
                return 0

    def dictionary_update(self, dataArray, id2vocab_file, vocab_file, tensor_file):

        if not os.path.exists(vocab_file):
            self.word2id = dict()
            self.vocabulary_size = -1
        else:
            with open(vocab_file, 'rb') as f:
                self.vocab = cPickle.load(f)
            self.vocabulary_size = len(self.vocab)
            self.word2id = dict(zip(self.vocab.keys(), self.vocab.values()))
        for p in dataArray:
            for word in p:
                # if not in dictionary, store index idCount+1
                if self.word2id.get(word) is None:
                    self.vocabulary_size += 1
                    self.word2id[word] = self.vocabulary_size
        # put char into dictionary random sort
        self.word2id = dict(zip(self.word2id.keys(), self.word2id.values()))
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.word2id, f)
        with open(id2vocab_file, 'wb') as f:
            cPickle.dump(self.id2word, f)
        # convert data sets to number
        for i in dataArray:
            self.valueArray.append(list(map(self.word2id.get, i)))
        self.tensor = np.array(self.valueArray)
        np.save(tensor_file, self.tensor)


if __name__ == '__main__':
    dataLoader()
