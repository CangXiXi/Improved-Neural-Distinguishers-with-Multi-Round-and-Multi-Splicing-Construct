import speck48_MRMSD as sp
import pickle

import numpy as np

from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate,dropout
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, AveragePooling1D,Conv2D, Conv1D, MaxPooling1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


bs = 5000
def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def create_model(depth):
    num_filters=48
    num_outputs=1
    d=64
    ks=3
    reg_param=0.0001
    inp = Input(shape=(4608,))
    rs = Reshape((48, 96))(inp)
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
    x_round=num_rounds
    data_x,data_y,ks_back=sp.make_train_data(train_num,x_round,diff)
    x_val,y_val=sp.make_train_val_data(val_num,x_round,diff,ks_back)
    seed=199847
    np.random.seed(seed)
    model=create_model(depth)
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    check = make_checkpoint(wdir+'speck48_M_hou2'+'_best_r'+str(num_rounds)+'.h5');
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));
    history = model.fit(data_x, data_y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data=(x_val, y_val), callbacks=[lr, check]);
    pickle.dump(history.history, open(wdir+'speck48_M_hou2'+'r'+str(num_rounds)+'.p', 'wb'));
    np.save(wdir+'speck48_M_hou2'+'r'+str(num_rounds)+'val_acc'+'.npy', history.history['val_acc']);
    np.save(wdir+'speck48_M_hou2'+'r'+str(num_rounds)+'val_loss'+'.npy', history.history['val_loss']);
    np.save(wdir+'speck48_M_hou2'+'r'+str(num_rounds)+'loss'+'.npy', history.history['loss']);
    np.save(wdir+'speck48_M_hou2'+'r'+str(num_rounds)+'acc'+'.npy', history.history['acc']);
    np.save(wdir + 'speck48_M_hou2_ksback_' + 'r' + str(num_rounds) + '.npy', ks_back)
    print("Best validation accuracy: ", np.max(history.history['val_acc']));
    model.save(wdir+'speck48_M_hou2'+'r'+str(num_rounds) + '.h5')
