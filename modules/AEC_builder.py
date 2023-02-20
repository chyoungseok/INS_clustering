from tensorflow.keras.layers import Input, add, Add, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import PReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers

class build_functions():
    @staticmethod
    def _conv_pool_relu(**conv_params):
        '''
        - create a layer consisting of Conv2D + ReLU(or PReLU) + MaxPooling2D
        - usage
          _conv_layer = _conv_pool_relu(params)(iput)
        '''
        weight_decay = 0.0005
        
        def f(input_img):
            input_img = Conv2D(filters=conv_params['nb_filter'],
                            input_shape=conv_params['input_shape'],
                            kernel_size=conv_params.setdefault('kernel_size', (3, 3)),
                            strides=conv_params.setdefault('strides', (1, 1)),
                            kernel_regularizer=regularizers.l2(l=weight_decay),
                            padding=conv_params.setdefault('padding', 'valid'))(input_img)
            # Batch Normalization?
            # ReLU or PReLU?
            # what is the proper arrangement of these functions?
            # input_img = PReLU()(input_img)
            return input_img
        return f


class Encoder():
    def __init__(self, **params):
        '''
        params
         - input_shape: shape of input image
         - nb_layer: number of convolutional-hidden layers
         
        '''
        self.input_img = Input(shape=params['input_shape'])
        self.nb_layer = params.setdefault('nb_layer', 3)
        
    def build(self):
        for i in range(self.nb_layer):
            conv = build_functions._conv_pool_relu(filters=) # nb_layer에 따라서 filter 수, kernel_size 등을 생성하는 계산식을 도출해서 적용해야 함
        
            
        
        

class Bottleneck():
    pass

class Decoder_Upsampling():
    pass

class Decoder_Deconv():
    pass
