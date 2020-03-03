import os, sys, time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import wraps, reduce

def get_data(annotation_line, input_shape, random=False):
    image = Image.open(annotation_line[0])
    iw, ih = image.size
    h, w = input_shape

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255.

        # correct labels
        return image_data

from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, UpSampling2D, Concatenate, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.applications.vgg16 import VGG16

# # Classifier block
# pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
# flatten = Flatten()(pool)
# predictions_g = Dense(units=2, kernel_initializer="he_normal", use_bias=False,
#                         kernel_regularizer=l2(0.0005), activation="softmax",
#                         name="pred_gender")(flatten)
# predictions_a = Dense(units=101, kernel_initializer="he_normal", use_bias=False,
#                         kernel_regularizer=l2(0.0005), activation="softmax",
#                         name="pred_age")(flatten)
# model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def create_model(inputs,n_classes={'emotion':6,'age':101,'gender':2}):

    vgg16 = VGG16(input_tensor=inputs,weights='imagenet',include_top=False)
    f1 = vgg16.get_layer('block5_pool').output

    x = compose(
            DarknetConv2D(512, (1,1),name='block6_conv'),
            BatchNormalization(name='block6_bn'),
            LeakyReLU(alpha=0.1,name='block6_lrelu'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block6_pool'))(f1)

    y_e = compose(
            DarknetConv2D(n_classes['emotion'], (1,1),name='block7_conv'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block7_pool'),
            Flatten(name='block7_flat'))(x)

    y_g = compose(
            DarknetConv2D(256, (3,3),name='block8_conv1'),
            BatchNormalization(name='block8_bn'),
            LeakyReLU(alpha=0.1,name='block8_lrelu'),
            DarknetConv2D(n_classes['gender'], (1,1),name='block8_conv2'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block8_pool'),
            Flatten(name='block8_flat'))(x)
    

    f2 = vgg16.get_layer('block5_conv3').output

    x = compose(
            DarknetConv2D(512, (1,1),name='block9_conv1'),
            UpSampling2D(4,name='block9_upsample'))(x)
            
    x = Concatenate(name='block9_concat')([x,f2])

    y_a = compose(
            DarknetConv2D(256, (1,1),name='block9_conv2'),
            BatchNormalization(name='block9_bn'),
            LeakyReLU(alpha=0.1,name='block9_lrelu'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool1'),
            DarknetConv2D(256, (1,1),name='block9_conv3'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool2'),
            DarknetConv2D(n_classes['age'], (1,1),name='block9_conv4'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool3'),
            Flatten(name='block9_flat'))(x)


    return Model(inputs = inputs, outputs=[y_e,y_g, y_a])

image_input = Input(shape=(128,128, 3))

m = create_model(image_input)

print(m.summary())

for layer in m.layers:
    print(layer.name,layer.trainable)

for layer in m.layers:
    n_start = layer.name.split('_')[0]
    if  n_start in ['block9','block8']:
        layer.trainable = False

print("----------------updated-----------------------")

for layer in m.layers:
    print(layer.name,layer.trainable)
    