import xarray as xr
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from scipy.io import savemat
import os
from keras.layers import Activation, Conv2D
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)



# ###########params
HEIGHT = 80
WIDTH = 160
nbClass = 3
Optimizer = 'Adam'

class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input

        return out
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint,
        })
        return config



class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma*aaTa + input
        return out
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint,
        })
        return config


# #################test data set##################################
data_train = xr.open_dataset('../data/eddy_sla.nc')
data_train = data_train.transpose('day', 'lat', 'lon')

data_label = xr.open_dataset('../data/eddy_label.nc')
data_label = data_label.transpose('day', 'lat', 'lon')
x_train1 = data_train.sla.data

data_train2 = xr.open_dataset('../data/ssu_bsc.nc')
x_train2 = data_train2.ssu.data
data_train3 = xr.open_dataset('../data/ssv_bsc.nc')
x_train3 = data_train3.ssv.data
m, n, k = np.shape(x_train1)
x_train1 = x_train1.reshape(m, n, k, 1)
x_train2 = x_train2.reshape(m, n, k, 1)
x_train3 = x_train3.reshape(m, n, k, 1)

x_train = np.concatenate((x_train1, x_train2,x_train3), axis=3)
x_label = data_label.label.data[:]
mt,nt,kt,qt = np.shape(x_train)
x_train_old = np.zeros_like(x_train)
for i in range(3):

    pixels = x_train[:, :, :, i].ravel()  # falatten
    x_max = max(pixels)
    x_min = min(pixels)
    x_train_old[:, :, :, i] = (x_train[:, :, :, i]-x_min)/(x_max-x_min)


x_train_old = x_train_old[6574:, :, :, :]
x_train_old = np.nan_to_num(x_train_old)
x_label_old = np.nan_to_num(x_label)
x_label_old = x_label_old[6574:, :, :]

x_train_new, x_label_new = x_train_old, x_label_old
label_train = np.reshape(x_label_new, (len(x_train_new), HEIGHT*WIDTH, 1))
x_train_label = np.zeros((len(x_train_new), HEIGHT*WIDTH, nbClass))
for kk in range(len(x_train_new)): # one-hot
    x_train_label[kk, :, :] = to_categorical(label_train[kk, :, :], nbClass)

# #########################loss and mean iou#############################################
smooth = 1e-5
def dice_coef_anti(y_true, y_pred):
    y_true_anti = y_true[:, :, 2]
    y_pred_anti = y_pred[:, :, 2]
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti) + K.sum(y_pred_anti) + smooth)


def dice_coef_cyc(y_true, y_pred):
    y_true_cyc = y_true[:, :, 1]
    y_pred_cyc = y_pred[:, :, 1]
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)


def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[:, :, 0]
    y_pred_nn = y_pred[:, :, 0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)


def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred)) / 3.


def dice_coef_loss(y_true, y_pred):
    return 1 - mean_dice_coef(y_true, y_pred)


def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def mean_iou( y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels

eddynet = models.load_model('../output/model_save/eddydan_suv.h5',custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef_anti':dice_coef_anti,'dice_coef_cyc':dice_coef_cyc,'mean_dice_coef':mean_dice_coef,'mean_iou':mean_iou,'InstanceNormalization': InstanceNormalization,'PAM':PAM,'CAM':CAM,'Zeros':tf.zeros_initializer(),'dice_coef_nn':dice_coef_nn})

eddynet.compile(optimizer=Optimizer, loss=dice_coef_loss,
                metrics=[dice_coef_anti, dice_coef_cyc,dice_coef_nn, mean_dice_coef, mean_iou, 'accuracy'])
tf.config.experimental_run_functions_eagerly(True)
loss = eddynet.evaluate(x_train_new, x_train_label, batch_size=5)
print('/nloss', loss)
eddy = eddynet.predict(x_train_old, batch_size=5)
eddy = eddy.argmax(2).reshape(len(x_train_new), HEIGHT, WIDTH)
savemat('../output/detection_result/eddydan_suv.mat', {'eddy': eddy,'evaluate':loss})

