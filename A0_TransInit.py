# /**
#  * @author [Zhongya]
#  * @email [zhongya66@whu.edu.cn]
#  * @create date 2021-09-7 11:00:30
#  * @modify date 2021-09-11 14:23:06
#  * @desc [description]
#  */
# -*- coding: utf-8 -*-
## for tf2
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, TimeDistributed, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
import scipy.io as scio

def Complex_Layer(x):
    return tf.complex(x[:,0],x[:,1])

## parameter
epoch            = 4000
batch_size       = 128
ratio            = 0.8
output_dim       = 2

train_mode = 1

## load training data
Extra  = 'mmWave'
prefix = '_'+''
Order_bit = 6

name_Tx_bit      = 'Init/BitData_QAM'+str(2**Order_bit)+'.txt'
name_Tx_sym_I    = 'Init/QAM'+str(2**Order_bit)+'_I.txt'
name_Tx_sym_Q    = 'Init/QAM'+str(2**Order_bit)+'_Q.txt'
origindata_path  = '''./'''
output_path      = 'NN/'
x_Tx_bit         = np.loadtxt(origindata_path+name_Tx_bit)
x_Tx_sym_I    = np.loadtxt(origindata_path+name_Tx_sym_I)[:,np.newaxis]
x_Tx_sym_Q    = np.loadtxt(origindata_path+name_Tx_sym_Q)[:,np.newaxis]
# x_Tx_sym_IQ   = np.concatenate((x_Tx_sym_I,x_Tx_sym_Q),axis = 1)
x_Tx_sym_IQ   = x_Tx_sym_I+x_Tx_sym_Q*1j

x_train_bit_blk  = x_Tx_bit
y_train_sym_blk  = x_Tx_sym_IQ

## optimizer for AE
optimizer = 'Adam'#keras.optimizers.SGD(lr=0.01, momentum=0.3, decay=0.0, nesterov=False)
loss      = 'categorical_crossentropy'
seed      = None
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)

checkpoint1 = ModelCheckpoint(filepath = './models/TransInit_'+Extra+prefix+str(Order_bit)+'bits'+'.hdf5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

## build the AE
if train_mode==1:
    # transmitter_bit    
    input_bit = Input(shape = (Order_bit,))
    y1_bit = Dense(Order_bit*4,activation = 'tanh')(input_bit)
    y2_bit = Dense(Order_bit*4,activation = 'tanh')(y1_bit)
    tx = Dense(output_dim,activation = 'tanh')(y2_bit)
    tx_IQ = Lambda(lambda input: Complex_Layer(input))(tx)
    Trans_model_bit = Model(inputs=input_bit,outputs = tx_IQ,name = 'Transmitter')
    
    Trans_model_bit.summary()

    Trans_model_bit.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=[])

    HisI = Trans_model_bit.fit([x_train_bit_blk], [y_train_sym_blk],
            epochs=epoch,
            batch_size=batch_size,
            callbacks=[checkpoint1])

    # fig, ax1 = plt.subplots()
    # ax1.plot(HisI.epoch,HisI.history['loss'],'r',label="Train loss ")
    # plt.plot(HisI.epoch,HisI.history['val_loss'],'b',label="Validation loss")
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('loss')
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # plt.show()
elif train_mode==2:
    Trans_model_bit = tf.keras.models.load_model('./models/TransInit_'+Extra+prefix+str(Order_bit)+'bits'+'.hdf5')
    Trans_model_bit.fit([x_train_bit_blk], [y_train_sym_blk],
            epochs=epoch,
            batch_size=batch_size,
            callbacks=[checkpoint1])
    

Trans_model_bit = keras.models.load_model('./models/TransInit_'+Extra+prefix+str(Order_bit)+'bits'+'.hdf5')
# Trans_model_bit = Trans_model_bit_block.get_layer('time_distributed').layer

## save data
Encoder_out_block = Trans_model_bit.predict(x_Tx_bit,batch_size=batch_size)

# IQ separate
Encoder_out_block_I = np.real(Encoder_out_block)[:,np.newaxis]
Encoder_out_block_Q = np.imag(Encoder_out_block)[:,np.newaxis]
Encoder_out_IQ_sep = np.concatenate([Encoder_out_block_I,Encoder_out_block_Q],axis = 1)
np.savetxt((origindata_path+output_path+'Encoder_Init_dataout_'+Extra+prefix+str(Order_bit)+'bits'+'.txt'),Encoder_out_IQ_sep,fmt='%1.7e')


print('seed =',seed,'; All saved!','; Order_bit =',Order_bit)