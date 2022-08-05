# coding: utf-8

import tensorflow as tf
from tensorflow.keras import models, layers,optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import xarray as xr
import numpy as np
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CUDNN_DETERMINISTIC']='0'

config= tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 #0.7为参数，含义为分配给gpu-memory的比例
sess=tf.compat.v1.Session(config=config)



# input data
HEIGHT = 80
WIDTH = 160
nbClass = 3
INPUT_CHANNEL = 3 # sla and geostrophic velocity
OUTPUT_MASK_CHANNEL = 3
# network structure
FILTER_NUM = 64 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 2 # size of upsampling filters
SE_RATIO = 16.   # reduction ratio of SE block



def expend_as(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(x, filter_size, size, dropout, batch_norm=True):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    axis = 3
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = InstanceNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = InstanceNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1),padding='same')(x)
    if batch_norm is True:
        shortcut = InstanceNormalization(axis=axis)(shortcut)

    res_path = layers.add([shortcut, conv])
    return res_path

def SE_block(x, out_dim, ratio, name, batch_norm=True):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    # x_s = layers.GlobalAveragePooling2D(data_format=None)(x)
    # # Excitation: bottom-up top-down FCs
    # if batch_norm:
    #     x_s = InstanceNormalization()(x_s)
    # x_e = layers.Dense(units=out_dim//ratio,kernel_regularizer=regularizers.l2(0.3),
    #             activity_regularizer=regularizers.l1(0.3))(x_s)
    # x_e = layers.Activation('tanh')(x_e)
    # if batch_norm:
    #     x_e = InstanceNormalization()(x_e)
    # x_e = layers.Dense(units=out_dim, kernel_regularizer=regularizers.l2(0.9),
    #             activity_regularizer=regularizers.l1(0.9))(x_e)
    # x_e = layers.Activation('softmax')(x_e)
    # x_e = layers.Reshape((1, 1, out_dim), name=name+'channel_weight')(x_e)
    # output = layers.multiply([x, x_e])
    # return output

    x_s = layers.GlobalAveragePooling2D(data_format=None)(x)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = InstanceNormalization()(x_s)
    x_e = layers.Dense(units=out_dim//ratio)(x_s)
    x_e = layers.Activation('relu')(x_e)
    if batch_norm:
        x_e = InstanceNormalization()(x_e)
    x_e = layers.Dense(units=out_dim)(x_e)
    x_e = layers.Activation('sigmoid')(x_e)
    x_e = layers.Reshape((1, 1, out_dim), name=name+'channel_weight')(x_e)
    result = layers.multiply([x, x_e])
    return result

def gating_signal(input, out_size, batch_norm=True):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape, name):
    """
    self gated attention, attention mechanism on spatial dimension
    :param x: input feature map
    :param gating: gate signal, feature map from the lower layer
    :param inter_shape: intermedium channle numer
    :param name: name of attention layer, for output
    :return: attention weighted on spatial dimension feature map
    """

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16
    # upsample_g = layers.UpSampling2D(size=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
    #                                  data_format="channels_last")(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('tanh')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),
                                       name=name+'_weight')(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])


    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = InstanceNormalization()(result)
    return result_bn


def Attention_ResUNet_PA(dropout_rate=0.6, batch_norm=True):
    '''
    Rsidual UNet construction, with attention gate1
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: nets
    '''
    # input data
    # dimension of the image depth
    inputs = layers.Input((HEIGHT, WIDTH, INPUT_CHANNEL), dtype=tf.float32)
    # inputs = HyperDenseModel()
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers

    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # channel attention block
    se_conv_16 = SE_block(conv_16, out_dim=8*FILTER_NUM, ratio=SE_RATIO, name='att_16')
    # spatial attention block
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(se_conv_16, gating_16, 8*FILTER_NUM, name='att_16')
    # attention re-weight & concatenate
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 7
    # channel attention block
    se_conv_32 = SE_block(conv_32, out_dim=4*FILTER_NUM, ratio=SE_RATIO, name='att_32')
    # spatial attention block
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(se_conv_32, gating_32, 4*FILTER_NUM, name='att_32')
    # attention re-weight & concatenate
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 8
    # channel attention block
    se_conv_64 = SE_block(conv_64, out_dim=2*FILTER_NUM, ratio=SE_RATIO, name='att_64')
    # spatial attention block
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(se_conv_64, gating_64, 2*FILTER_NUM, name='att_64')
    # attention re-weight & concatenate
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 9
    # channel attention block
    se_conv_128 = SE_block(conv_128, out_dim=FILTER_NUM, ratio=SE_RATIO, name='att_128')
    # spatial attention block
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    # attention re-weight & concatenate
    att_128 = attention_block(se_conv_128, gating_128, FILTER_NUM, name='att_128')
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = layers.Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    # conv_final = InstanceNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)
    # print(conv_final)
    conv_final = layers.Reshape((HEIGHT * WIDTH, OUTPUT_MASK_CHANNEL))(conv_final) # The output corresponds to one-hot

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionSEResUNet")
    return model


# ###################DATASET############################ #

data_train = xr.open_dataset('../../data/eddy_sla.nc')
data_train = data_train.transpose('day', 'lat', 'lon')

data_label = xr.open_dataset('../../data/sla_concat/eddy_label.nc')
data_label = data_label.transpose('day', 'lat', 'lon')
x_train1 = data_train.sla.data
data_train2 = xr.open_dataset('E:/term/eddy_matlab/ssu_bsc.nc')
x_train2 = data_train2.ssu.data
data_train3 = xr.open_dataset('E:/term/eddy_matlab/ssv_bsc.nc')
x_train3 = data_train3.ssv.data
m, n, k = np.shape(x_train1)
x_train1 = x_train1.reshape(m, n, k, 1)
x_train2 = x_train2.reshape(m, n, k, 1)
x_train3 = x_train3.reshape(m, n, k, 1)

x_train = np.concatenate((x_train1, x_train2,x_train3), axis=3)
x_label = data_label.label.data[:]
mt,nt,kt,qt = np.shape(x_train)
x_train_old = np.zeros_like(x_train)
for i in range(3): # Normalization
    pixels = x_train[:, :, :, i].ravel()  # flatten
    x_max = max(pixels)
    x_min = min(pixels)
    x_train_old[:, :, :, i] = (x_train[:, :, :, i]-x_min)/(x_max-x_min)



x_train_old = x_train_old[:6573, :, :, :]
x_train_old = np.nan_to_num(x_train_old)
x_label_old = np.nan_to_num(x_label)
x_label_old = x_label_old[:6573, :, :]

from sklearn.utils import shuffle
x_train_new, x_label_new = shuffle(x_train_old, x_label_old)

label_train = np.reshape(x_label_new, (len(x_train_new), HEIGHT*WIDTH, 1))
x_train_label = np.zeros((len(x_train_new), HEIGHT*WIDTH, nbClass))
for kk in range(len(x_train_new)):
    x_train_label[kk, :, :] = to_categorical(label_train[kk, :, :], nbClass)


# ######################################## metrics #############################################
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
# ################################## loss ###########################################

def dice_coef_anti(y_true, y_pred):
    smooth = 1e-5
    y_true_anti = K.flatten(y_true[:, :, 2])
    y_pred_anti = K.flatten(y_pred[:, :, 2])
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti+smooth) / (K.sum(y_true_anti) + K.sum(y_pred_anti) + smooth)
#
#
def dice_coef_cyc(y_true, y_pred):
    smooth = 1e-5
    y_true_cyc = K.flatten(y_true[:, :, 1])
    y_pred_cyc = K.flatten(y_pred[:, :, 1])
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc+smooth ) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)
#
#
def dice_coef_nn(y_true, y_pred):
    smooth = 1e-5
    y_true_nn = K.flatten(y_true[:, :, 0])
    y_pred_nn = K.flatten(y_pred[:, :, 0])
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn +smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)
#
#
def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred)) / 3.

def dice_coef_loss(y_true, y_pred):
    return 1 - mean_dice_coef(y_true, y_pred)


# ########################################### COMPILE AND TRAIN #################################################
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, mode='min', restore_best_weights=True)
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='min',
                         min_delta=0.001, min_lr=1e-4, verbose=1)


eddynet = Attention_ResUNet_PA()

opt = optimizers.Adam(learning_rate=0.001, decay=1e-6)
eddynet.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef_anti, dice_coef_cyc,dice_coef_nn, mean_dice_coef, mean_iou, 'accuracy'])
tf.config.experimental_run_functions_eagerly(True)
eddynet.fit(x_train_new[:, :, :, :], x_train_label[:, :, :], epochs=100, batch_size=15, validation_split=0.3, callbacks=[early_stopping, reduce_lr])
eddynet.save('../output/model_save/eddyatt_suv.h5')
