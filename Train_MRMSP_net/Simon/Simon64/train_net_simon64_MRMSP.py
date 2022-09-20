import simon64_MRMSP as sm
import pickle

import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2


bs = 5000
def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);
def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
  return(res)
def create_model(depth):
    num_filters=sm.block_size()
    num_outputs=1
    d=64
    ks=3
    reg_param=0.0001
    inp = Input(shape=(4096,))
    rs = Reshape((16, 256))(inp)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(rs)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0;
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)

def train_model(train_num,val_num,num_rounds,num_epochs,depth,diff,wdir):
    data_x,data_y,ks_back=sm.make_train_data(train_num,num_rounds,diff)
    x_val,y_val=sm.make_train_val_data(val_num,num_rounds,diff,ks_back)
    seed=199847
    np.random.seed(seed)
    model=create_model(depth)
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    check = make_checkpoint(wdir+'simon64_M_chen2'+'_best_r'+str(num_rounds)+'.h5');
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));
    history = model.fit(data_x, data_y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data=(x_val, y_val), callbacks=[lr, check]);
    pickle.dump(history.history, open(wdir+'simon64_M_chen2'+'r'+str(num_rounds)+'.p', 'wb'));
    np.save(wdir+'simon64_M_chen2'+'r'+str(num_rounds)+'val_acc'+'.npy', history.history['val_acc']);
    np.save(wdir+'simon64_M_chen2'+'r'+str(num_rounds)+'val_loss'+'.npy', history.history['val_loss']);
    np.save(wdir+'simon64_M_chen2'+'r'+str(num_rounds)+'loss'+'.npy', history.history['loss']);
    np.save(wdir+'simon64_M_chen2'+'r'+str(num_rounds)+'acc'+'.npy', history.history['acc']);
    np.save(wdir + 'simon64_M_chen2_ksback_' + 'r' + str(num_rounds) + '.npy', ks_back)
    print("Best validation accuracy: ", np.max(history.history['val_acc']));
    model.save(wdir+'simon64_M_chen2'+'r'+str(num_rounds) + '.h5')
