from data import *
from keras.optimizers import Adam
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from metric import PSNR,psnr
from my_model import my_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
noise_level = 1
train_file = 'DATA/train/'
train_mask_file = 'DATA/train_label/'
txt_file = 'DATA/txt_file/train.txt'
valid_file = 'DATA/valid/'
valid_mask_file = 'DATA/valid_label/'
valid_txt_file = 'DATA/txt_file/valid.txt'
myGene = BatchGenerator(train_file,train_mask_file,txt_file,txt_file)
myvalidGene = BatchGenerator(valid_file,valid_mask_file,valid_txt_file,valid_txt_file)
model = my_model((64,64,64,1))
model.summary()
model.compile(optimizer = Adam(lr = 1e-3), loss = 'mse', metrics = ['mse',PSNR])
model_checkpoint1 = ModelCheckpoint('weights.{epoch:02d}-{loss:.6f}.hdf5',monitor='loss',verbose = 1,save_best_only=True,save_weights_only=True,mode='auto',period=1)
early_stop = EarlyStopping(monitor='loss',patience=20)
model.fit_generator(myGene,steps_per_epoch=6400,epochs=20,validation_data=myvalidGene,validation_steps = 1280,callbacks=[model_checkpoint1,early_stop])
