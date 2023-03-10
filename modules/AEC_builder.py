from tensorflow.keras.layers import Input, add, Add, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.layers import PReLU, BatchNormalization, Flatten, Dense, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import numpy as np

def act_function(input, selection):
    if selection == 'ReLU':
        f = ReLU()(input)
    elif selection == 'PReLU':
        f = PReLU()(input)
    return f

def check_shape(layer, nb_layer, conv_kernel, conv_stride, pool_kernel, pool_stride, padding):
    shape = layer.shape
    shape_idx = 1, 2
    
    for i, idx in enumerate(shape_idx):
        # row, column에 대한 반복
        temp_shape = shape[idx]
        iter = nb_layer
        
        while iter > 0: # number of layer 만큼 반복
            if padding == 'valid':
                # convolutional output shape
                temp_shape = np.ceil(float(temp_shape - conv_kernel[i] + 1) / float(conv_stride[i]))
                # pooling output shape
                temp_shape = np.ceil(float(temp_shape - pool_kernel[i] + 1) / float(pool_stride[i]))
                
            elif padding == 'same':
                # convolutional output shape
                temp_shape = np.ceil(float(temp_shape) / float(conv_stride[i]))
                # pooling output shape
                temp_shape = np.ceil(float(temp_shape) / float(pool_stride[i]))
            
            if temp_shape < 1:
                return False
            
            iter -= 1
    
    return True


class build_functions():
    @staticmethod
    def _conv_bn_relu_pool(**conv_params):
        '''
        - create a layer consisting of Conv2D + BatchNormalization + ReLU(or PReLU) + MaxPooling2D
        - usage
          _conv_layer = _conv_pool_relu(params)(iput)
        '''
        weight_decay = 0.0005
        
        def f(input_img):
            if conv_params['is_first_layer']:
                input_img = Conv2D(filters=conv_params['nb_filter'],
                                input_shape=conv_params['input_shape'],
                                kernel_size=conv_params['kernel_size'],
                                strides=conv_params['strides'],
                                kernel_regularizer=regularizers.l2(l=weight_decay),
                                padding=conv_params['padding'])(input_img)
            else:
                input_img = Conv2D(filters=conv_params['nb_filter'],
                                kernel_size=conv_params['kernel_size'],
                                strides=conv_params['strides'],
                                kernel_regularizer=regularizers.l2(l=weight_decay),
                                padding=conv_params['padding'])(input_img)
                
            input_img = BatchNormalization()(input_img)
            input_img = act_function(input=input_img, selection=conv_params['act_selection'])
            input_img = MaxPooling2D(pool_size=conv_params['pool_size'], strides=conv_params['pool_strides'], padding=conv_params['padding'])(input_img)    
            return input_img
        return f

    @staticmethod
    def _conv_bn_relu_upsample(**conv_params):
        '''
        - create a layer consisting of COnv2D + BatchNormalization + ReLU(or PReLU) + UpSampling2D
        - usage
          _conv_upsample_layer = _conv_bn_relut_upsample(params)(input)
        '''
        weight_decay = 0.0005
        
        def f(input_img):
            input_img = Conv2D(filters=conv_params['nb_filter'],
                            kernel_size=conv_params['kernel_size'],
                            strides=conv_params['strides'],
                            kernel_regularizer=regularizers.l2(l=weight_decay),
                            padding=conv_params['padding'])(input_img)

            input_img = BatchNormalization()(input_img)
            input_img = act_function(input=input_img, selection=conv_params['act_selection'])
            input_img = UpSampling2D(size=conv_params['pool_size'])(input_img)
            return input_img
        return f

    @staticmethod
    def _deconv_bn_relu(**conv_params):
        weight_decay = 0.0005
        
        def f(input_img):
            input_img = Conv2DTranspose(filters=conv_params['nb_filter'],
                                        kernel_size=conv_params['kernel_size'],
                                        strides=conv_params['strides'],
                                        kernel_regularizer=regularizers.l2(l=weight_decay),
                                        padding=conv_params['padding'])(input_img)
            input_img = BatchNormalization()(input_img)
            input_img = act_function(input=input_img, selection=conv_params['act_selection'])
            return input_img
        return f


class Encoder():
    def __init__(self, **params):
        '''
        params
         - input_shape: shape of input image
         - nb_layer: number of convolutional-hidden layers
        '''
        self.params = params
        self.input_img = Input(shape=params.setdefault('input_shape', (2000,16,1)))
        self.nb_layer = params.setdefault('nb_layer', 3)
        
        self.input_shape = params.setdefault('input_shape', (2000,16,1))
        self.kernel_size = params.setdefault('kernel_size', (10,3))
        self.strides = params.setdefault('strides', (1,1))
        self.padding = params.setdefault('padding', 'valid')
        self.act_selection = params.setdefault('act_selection', 'ReLU')
        self.pool_size = params.setdefault('pool_size', (5,3))
        self.pool_strides = params.setdefault('pool_strides', (2,2))            
        
    def build(self):
        # input이 layer를 통과 했을 때, 연산 결과, shape 차원에서 오류가 발생하지 않는지 확인
        self.error_shape = False
        if not(check_shape(layer=self.input_img, nb_layer=self.nb_layer, 
                           conv_kernel=self.kernel_size, conv_stride=self.strides,
                           pool_kernel=self.pool_size, pool_stride=self.pool_strides,
                           padding=self.padding)):
            print('Consider the resulting shape !')
            self.error_shape = True
            return
        
        conv = self.input_img
        nb_filter = 8
        for i in range(self.nb_layer):
            conv = build_functions._conv_bn_relu_pool(nb_filter = nb_filter,
                                                      is_first_layer = (i==0),
                                                      input_shape = self.input_shape,
                                                      kernel_size = self.kernel_size,
                                                      strides = self.strides,
                                                      padding = self.padding,
                                                      act_selection = self.act_selection,
                                                      pool_size = self.pool_size,
                                                      pool_strides=self.pool_strides
                                                      )(conv)              
            nb_filter *= 2
        model = Model(inputs=self.input_img, outputs=conv)
        self.model = model       

class BottleNeck():
    def __init__(self, **params):
        self._encoder = params['_encoder'] # model of encoder
        self._encoder_shape = self._encoder.output.shape # shape of output from encoder
        self.vector_len = params.setdefault('vector_len', 10) # length of a latent vector
        self.act_selection = params.setdefault('act_selection', 'ReLU')
        self.use_DENSE_OR_GAP = params.setdefault('use_DENSE_OR_GAP', 'DENSE')
        
    
    def build(self):
        embedding = self._encoder.output
        
        if self.use_DENSE_OR_GAP == 'DENSE':
            embedding = Flatten()(embedding)
            embedding = Dense(units=self._encoder_shape[1]*self._encoder_shape[2]*self._encoder_shape[3])(embedding)
            embedding = BatchNormalization()(embedding)
            embedding = act_function(input=embedding, selection=self.act_selection)
            embedding = Dense(units=self.vector_len, name='embedding')(embedding)
            
            decoder = embedding
            decoder = Dense(units=(self._encoder_shape[1])*self._encoder_shape[2]*self._encoder_shape[3])(decoder)
            decoder = Reshape((self._encoder_shape[1], self._encoder_shape[2], self._encoder_shape[3]))(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = act_function(input=decoder, selection=self.act_selection)

            
        elif self.use_DENSE_OR_GAP == 'GAP':
            embedding = Dense(units=self.vector_len)(embedding)
            embedding = GlobalAveragePooling2D(name='embedding')(embedding)
            
            decoder = embedding
            decoder = Dense(units=self._encoder_shape[1]*self._encoder_shape[2]*self.vector_len)(decoder)
            decoder = Reshape((self._encoder_shape[1], self._encoder_shape[2], self.vector_len))(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = act_function(input=decoder, selection=self.act_selection)
            
            decoder = Dense(units=self._encoder_shape[3])(decoder)
            decoder = Reshape((self._encoder_shape[1], self._encoder_shape[2], self._encoder_shape[3]))(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = act_function(input=decoder, selection=self.act_selection)            
            
        
        self.model = Model(inputs=self._encoder.input, outputs=embedding)
        self.decoder = Model(inputs=self._encoder.input, outputs=decoder)

class Decoder():
    def __init__(self, **params):
        self._bottleneck = params['_bottleneck'] # model of bottleneck (as decoder)\
        self.use_UPSAMPLE_OR_DECONV = params['use_UPSAMPLE_OR_DECONV']

        self.nb_layer = params.setdefault('nb_layer', 3)
        self.kernel_size = params.setdefault('kernel_size', (10,3))
        self.strides = params.setdefault('strides', (1,1)) # UPSAMPLE 일때는 convolution의 strides (1,1), DECONV 일떄는 Conv2DTranspose의 strides (2,2)
        self.padding = params.setdefault('padding', 'valid')
        self.act_selection = params.setdefault('act_selection', 'ReLU')
        self.pool_size = params.setdefault('pool_size', (2,2)) # for Upsampling
    
    
    def build(self):
        decoder = self._bottleneck.output
        
        for i in range(self.nb_layer):
            nb_filter = (4**self.nb_layer)/(4**(i+1))
            if self.use_UPSAMPLE_OR_DECONV == 'UPSAMPLE':
                decoder = build_functions._conv_bn_relu_upsample(nb_filter = nb_filter,
                                                                kernel_size = self.kernel_size,
                                                                strides = self.strides,
                                                                padding = self.padding,
                                                                act_selection = self.act_selection,
                                                                pool_size = self.pool_size
                                                                )(decoder)
            elif self.use_UPSAMPLE_OR_DECONV == 'DECONV':
                decoder = build_functions._deconv_bn_relu(nb_filter = nb_filter,
                                                        kernel_size = self.kernel_size,
                                                        strides = self.strides,
                                                        padding = self.padding,
                                                        act_selection = self.act_selection
                                                        )(decoder)
        
        self.model = Model(inputs=self._bottleneck.input, outputs=decoder)

def ConvolutionalAutoencoder_v05(vector_len=30):
    weight_decay=0.0005

    model = Sequential()
    model.add(Conv2D(filters=24,
                        input_shape=(2000, 16, 1),
                        kernel_size=(10, 2),
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(l=weight_decay),
                        padding='valid', name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv2D(filters=48,
                        kernel_size=(10, 2),
                        strides=(2, 1),
                        kernel_regularizer=regularizers.l2(l=weight_decay),
                        padding='valid', name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool2D(pool_size=(5, 2),
                        strides=(2, 1), name="Pool1"))

    model.add(Flatten())
    model.add(Dense(units=vector_len, name='embedding'))

    model.add(Dense(units=494*14*48, activation='relu'))

    model.add(Reshape((494,14,48)))

    model.add(Conv2DTranspose(filters=24,
                                kernel_size=(10, 2),
                                kernel_regularizer=regularizers.l2(
                                    l=weight_decay),
                                strides=(2, 1), name="Deconv1", padding='valid'))
    model.add(BatchNormalization(name="BN3"))
    model.add(PReLU(name="PReLU3"))
    model.add(Conv2DTranspose(filters=1,
                                kernel_size=(10, 2),
                                kernel_regularizer=regularizers.l2(
                                    l=weight_decay),
                                strides=(2, 1), name="Deconv2", padding='valid'))
    model.add(BatchNormalization(name="BN4"))
    
    return model


def ConvolutionalAutoencoder_v06(vector_len=30):
    weight_decay=0.0005
    padding = 'same'

    model = Sequential()
    model.add(Conv2D(filters=24,
                        input_shape=(2000, 16, 1),
                        kernel_size=(10, 2),
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(l=weight_decay),
                        padding=padding, name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv2D(filters=48,
                        kernel_size=(10, 2),
                        strides=(2, 2),
                        kernel_regularizer=regularizers.l2(l=weight_decay),
                        padding=padding, name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool2D(pool_size=(5, 2),
                        strides=(2, 2), name="Pool1", padding=padding))

    model.add(Flatten())
    model.add(Dense(units=vector_len, name='embedding'))

    model.add(Dense(units=500*4*48, activation='relu'))

    model.add(Reshape((500,4,48)))

    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2DTranspose(filters=24,
                                kernel_size=(10, 2),
                                kernel_regularizer=regularizers.l2(
                                    l=weight_decay),
                                strides=(2, 2), name="Deconv1", padding=padding))
    model.add(BatchNormalization(name="BN3"))
    model.add(PReLU(name="PReLU3"))
    model.add(Conv2DTranspose(filters=1,
                                kernel_size=(10, 2),
                                kernel_regularizer=regularizers.l2(
                                    l=weight_decay),
                                strides=(1, 1), name="Deconv2", padding=padding))
    model.add(BatchNormalization(name="BN4"))
    
    return model