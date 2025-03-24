import numpy as np
from PIL import Image
import tensorflow as tf
import os
from keras import backend as K
from keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Reshape, Masking, Lambda, Permute
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from imp import reload
import net_ctc as net

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = net.dense_cnn(input, nclass)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1]
    return dic

class random_uniform_num():
    """均匀随机，确保每轮每个只出现一次"""
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if self.index + batchsize > self.total:
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize
        return r_n

def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float64)
    labels = np.ones((batchsize, maxlabellength)) * 10000
    input_length = np.zeros((batchsize, 1))
    label_length = np.zeros((batchsize, 1))

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5
            x[i] = np.expand_dims(img, axis=2)
            str_ = image_label[j]
            label_length[i] = len(str_)
            if len(str_) < 0:
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str_)] = [int(k) for k in str_]

        inputs = {
            'the_input': x,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros((batchsize,))}
        yield (inputs, outputs)


if __name__ == '__main__':

    img_height = 32
    img_width = 200
    batch_size = 128
    char_set = open('./Data/labels.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set[1:]])
    num_class = len(char_set)

    #K.set_session(get_session()) only in 1.X
    reload(net)
    model = get_model(img_height, num_class)

    train_set_file = './Data/data_3_train.txt'
    test_set_file = './Data/data_3_test.txt'
    train_loader = gen(train_set_file, './Data/train_imgs/', batchsize=batch_size, maxlabellength=num_class, imagesize=(img_height, img_width))
    test_loader = gen(test_set_file, './Data/test_imgs/', batchsize=batch_size, maxlabellength=num_class, imagesize=(img_height, img_width))

    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.1**epoch
    learning_rate = np.array([lr_schedule(i) for i in range(9)])
    changlr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    train_num_lines = sum(1 for line in open(train_set_file))
    test_num_lines = sum(1 for line in open(test_set_file))

    model.fit_generator(train_loader,
                        steps_per_epoch=train_num_lines // batch_size,
                        epochs=20,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=test_num_lines // batch_size,
                        callbacks=[checkpoint, earlystop, changlr])