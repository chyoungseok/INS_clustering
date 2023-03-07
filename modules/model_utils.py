import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

from modules import AEC_builder

temp_path = os.getcwd().split('Deep_clustering_v02')[0]
temp_path = os.path.join(temp_path, 'Deep_clustering_v02')  


def get_PARAMS_from_ExperimentInfo(temp_experiment):
    encoder_PARAMS = {
        'nb_layer' : temp_experiment['nb_layer'],
        'kernel_size' : (temp_experiment['kernel_width'], (temp_experiment['kernel_height'])),
        'strides' : (temp_experiment['strides_width'], temp_experiment['strides_height']),
        'padding' : temp_experiment['padding'],
        'act_selection' : temp_experiment['act_selection'],
        'pool_size' : (temp_experiment['max_pool_width'], temp_experiment['max_pool_height']),
        'pool_strides' : (temp_experiment['max_pool_strides_width'], temp_experiment['max_pool_strides_height'])   
    }
    
    bottleneck_PARAMS = {
        'vector_len' : temp_experiment['vector_len'],
        'act_selection' : temp_experiment['act_selection'],
        'use_DENSE_OR_GAP' : temp_experiment['use_DENSE_OR_GAP']
    }
    
    decoder_PARAMS = {
        'nb_layer' : temp_experiment['nb_layer'],
        'use_UPSAMPLE_OR_DECONV' : temp_experiment['use_UPSAMPLE_OR_DECONV'],
        'kernel_size' : (temp_experiment['decoder_kernel_width'], temp_experiment['decoder_kernel_height']),
        'strides' : (temp_experiment['decoder_strides_width'], temp_experiment['decoder_strides_height']),
        'padding' : temp_experiment['padding'],
        'act_selection' : temp_experiment['act_selection'],
        'pool_size' : (temp_experiment['decoder_pool_width'], temp_experiment['decoder_pool_height'])        
    }

    return encoder_PARAMS, bottleneck_PARAMS, decoder_PARAMS


def get_model_with_experiment_info(temp_experiment):
    encoder_PARAMS, bottleneck_PARAMS, decoder_PARAMS = get_PARAMS_from_ExperimentInfo(temp_experiment)
    _encoder = AEC_builder.Encoder(**encoder_PARAMS)
    _encoder.build()
    
    bottleneck_PARAMS['_encoder'] = _encoder.model
    _bottle_neck = AEC_builder.BottleNeck(**bottleneck_PARAMS)
    _bottle_neck.build()
    
    decoder_PARAMS['_bottleneck'] = _bottle_neck.decoder
    _decoder = AEC_builder.Decoder(**decoder_PARAMS)
    _decoder.build()
    
    return _decoder.model

def custom_corr(X_true, X_pred):
    corr_train = []
    for i in range(len(X_true)):
        temp_corr = np.corrcoef(K.flatten(tf.squeeze(X_true[i])),\
                                K.flatten(tf.squeeze(X_pred[i])))[0,1]
        corr_train.append(temp_corr)
        
    return np.mean(corr_train)

def corr_keras(X_true, X_pred):
    corr = tf.py_function(func=custom_corr, inp=[X_true, X_pred], Tout=tf.float32, name='corr')
    return corr

def model_initiation(model, is_lr_reducer, learning_rate):
    
    # Cofiguration of Optimizer
    if is_lr_reducer:
        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
    # Configuration of loss function
    
    # -------   
    # if you need to design custome loss, use this space
    get_custom_objects().update({'corr_keras':corr_keras})
    # -------
    metrics = ['MSE', corr_keras]
    loss = tf.keras.losses.MSE
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model
    
    
def train_model(X_train, X_val, test_name, is_lr_reducer, model, epochs, batch_size):  
    # callback_1
    path_weights = '%s/results/%s/weights.h5' % (temp_path, test_name)
    checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True, verbose=False, mode='min')
    callbacks = [checkpoint]

    # callback_2
    if is_lr_reducer:
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, cooldown=0, patience=10, min_lr=0.1e-8, verbose=True, mode='min')
        callbacks.append(lr_reducer)
        
    # Start training
    H = model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, callbacks=callbacks, batch_size=batch_size)
    
    # Plot acc_loss_plot and Save the result
    plot_acc_loss(H=H, path_save='%s/results/%s/acc_loss_plot.png' % (temp_path, test_name), epochs=epochs, test_name=test_name)
    return model



def eval_model(X_train, X_test, test_name):    
    model = load_model('%s/results/%s/weights.h5' % (temp_path, test_name))
    
    # 1. test MSE
    X_test_pred = model.predict(X_test)
    corr = custom_corr(X_test, X_test_pred)
    print("\ncorr of test set: {}".format(corr))
    
    # 2. Compare the image of input and output
    #       - two samples from both train and test data, respectively
    X_train_pred = model.predict(X_train)
    num_samples = 5
    save_path = os.path.join(temp_path, 'results', test_name, 'train example.png')
    f = plot_i_o_compare(X_train[0:num_samples], X_train_pred[0:num_samples], save_path=save_path)
    save_path = os.path.join(temp_path, 'results', test_name, 'test example.png')
    f = plot_i_o_compare(X_test[0:num_samples], X_test_pred[0:num_samples], save_path)


def plot_acc_loss(H, path_save, epochs, test_name):
    # input: output from model.fit()
    num_subplot=2

    plt.style.use("ggplot")
    plt.figure(figsize=(10*num_subplot,10))

    # MSE plot
    plt.subplot(1,num_subplot,1)
    plt.plot(np.arange(0, epochs), H.history["MSE"], label="train_mse")
    plt.plot(np.arange(0, epochs), H.history["val_MSE"], label="val_mse")
    plt.title('MSE_' + test_name, fontsize=20)
    plt.xlabel("Epoch #", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    
    # corr plot
    plt.subplot(1,num_subplot,2)
    plt.plot(np.arange(0, epochs), H.history["corr_keras"], label="train_corr")
    plt.plot(np.arange(0, epochs), H.history["val_corr_keras"], label="val_corr")
    plt.title('corr_' + test_name, fontsize=20)
    plt.xlabel("Epoch #", fontsize=20)
    plt.ylabel("corr", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    plt.savefig(path_save)
    
    
def plot_i_o_compare(inputs, outputs, save_path=None, title_input='input', title_output='output'):
    # plt.style.use("ggplot")

    if len(inputs.shape) < 4:
        inputs = inputs[np.newaxis, :, :, :]
        outputs = outputs[np.newaxis, :, :, :]

    num_samples = len(inputs)

    # plt.close()
    fig, axes = plt.subplots(2,1*num_samples,figsize=(10*num_samples,12));
    clim = (0.0, 1.5)

    for i in range(num_samples):
        corr = np.corrcoef(inputs[i].squeeze().flatten(),\
                            outputs[i].squeeze().flatten())[0,1]

        if num_samples == 1:
            ax_input = axes[0]
            ax_output = axes[1]
        else:
            ax_input = axes[0, i]
            ax_output = axes[1, i]


        im_input = ax_input.imshow(inputs[i].reshape(2000,16).transpose(1,0), cmap='hot', aspect=100, clim=clim);
        if num_samples != 1:
            title_input = 'input_%d' % (i+1)
            title_output = 'output_%d (%f)' % (i+1, corr)

        ax_input.set_title(title_input, fontsize=20); detail_modification(ax_input);
        fig.colorbar(im_input, shrink=0.5, ax=ax_input);

        im_output = ax_output.imshow(outputs[i].reshape(2000,16).transpose(1,0), cmap='hot', aspect=100, clim=clim);
        ax_output.set_title(title_output, fontsize=20); detail_modification(ax_output);
        fig.colorbar(im_output, shrink=0.5, ax=ax_output);      

    if save_path != None:
        plt.savefig(save_path)
        
        
def detail_modification(ax):
    ax.set_ylabel('frequency[Hz]', fontsize=15);
    ax.set_xlabel('sleep period', fontsize=15);
    ax.invert_yaxis();
    ax.yaxis.set_ticks(range(0,16));
    ax.yaxis.set_ticklabels([0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8, 5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, 37.5], fontsize=15);
    ax.grid(False);