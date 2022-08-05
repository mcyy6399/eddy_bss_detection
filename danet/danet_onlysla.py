from nets.danet import danet_resnet101
import numpy as np
from tensorflow.keras.callbacks import  EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import xarray as xr
from tensorflow.keras import backend as K
import tensorflow as tf


# network params
BatchSize = 5
NumChannels = 1
ImgHeight = 80
ImgWidth = 160
NumClass = 3

# training params
Optimizer = 'Adam'
NumEpochs = 100
Patience = 5

# ################################# dataset ############################################
data_train = xr.open_dataset('../data/eddy_sla.nc')
data_train = data_train.transpose('day', 'lat', 'lon')

data_label = xr.open_dataset('../data/eddy_label.nc')
data_label = data_label.transpose('day', 'lat', 'lon')
x_train1 = data_train.sla.data

m, n, k = np.shape(x_train1)
x_train1 = x_train1.reshape(m, n, k, 1)

x_label = data_label.label.data[:]
mt,nt,kt,qt = np.shape(x_train1)

# Normalization
pixels = x_train1.ravel()  # flatten
x_max = max(pixels)
x_min = min(pixels)
x_train_old = (x_train1-x_min)/(x_max-x_min)

x_train_old = x_train_old[:6573, :, :, :]
x_train_old = np.nan_to_num(x_train_old)

x_label_old = np.nan_to_num(x_label)
x_label_old = x_label_old[:6573, :, :]

from sklearn.utils import shuffle
x_train_new, x_label_new = shuffle(x_train_old, x_label_old)

label_train = np.reshape(x_label_new, (len(x_train_new), ImgHeight*ImgWidth, 1))
x_train_label = np.zeros((len(x_train_new),  ImgHeight*ImgWidth, NumClass))
for kk in range(len(x_train_new)):
    x_train_label[kk, :, :] = to_categorical(label_train[kk, :, :], NumClass)


# #################################### loss #########################
smooth = 1e-6
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

metric_list = [dice_coef_anti,dice_coef_cyc,dice_coef_nn,mean_dice_coef,'accuracy']

model = danet_resnet101(ImgHeight, ImgWidth, NumChannels, NumClass)

# ####################### train ####################################################
model.compile(optimizer=Optimizer, loss=dice_coef_loss, metrics=metric_list)

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=Patience)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2)
check_point_list = [ early_stop, reduce_lr]

tf.config.experimental_run_functions_eagerly(True)
model.fit(x_train_new, x_train_label, epochs=100, batch_size=BatchSize,validation_split=0.3, callbacks=check_point_list)
model.save('../output/model_save/eddydan_onlysla.h5')
