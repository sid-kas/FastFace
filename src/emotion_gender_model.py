import os, sys, time, random, math
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import wraps, reduce
import Augmentor
from keras.utils import Sequence, to_categorical

def rescale_image(image: Image, input_shape=(128,128)):
    iw, ih = image.size
    h, w = input_shape
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    # image_data = np.array(new_image)/255.

    return new_image

def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(input_image: Image):
        image = [input_image]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


def get_data(annotation_line, input_shape, random=False):
    image = Image.open(annotation_line[0])


class DataGenerator(Sequence):
    def __init__(self, df: pd.DataFrame,  batch_size=32, image_size=128, n_classes={'emotion':6,'age':101,'gender':2}):
        self.df = df
        self.n_classes = n_classes
        cols_needed = ['path','gender','emotion','age']
        for col in cols_needed:
            if col not in df.columns:
                self.n_classes[col] = False
                print("Expected to see "+ col + " but not found in the dataframe provided for face generator")
        
        self.image_num = self.df.shape[0]
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y_e = np.zeros((batch_size, 1), dtype=np.int32)
        y_g = np.zeros((batch_size, 1), dtype=np.int32)
        y_a = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            row_in_df = self.df.loc[sample_id,:]
            image = Image.open(row_in_df.path)
            x[i] = self.transform_image(rescale_image(image,input_shape=(self.image_size, self.image_size) ))

            if self.n_classes['emotion']:
                y_e[i] = row_in_df.emotion

            if self.n_classes['gender']:
                y_g[i] = row_in_df.gender

            if self.n_classes['age']:
                age = row_in_df.age
                age += math.floor(np.random.randn() * 2 + 0.5)
                y[i] = np.clip(age, 0, 100)
            
            
        
        y_emotion = to_categorical(y_e, self.n_classes['emotion']) if self.n_classes['emotion'] else None
        y_gender = to_categorical(y_e, self.n_classes['gender']) if self.n_classes['gender'] else None
        y_age = to_categorical(y_e, self.n_classes['age']) if self.n_classes['age'] else None
        
        return x, y_emotion , y_gender, y_age
        

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4), 'use_bias': False}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def create_model(inputs,n_classes={'emotion':6,'age':101,'gender':2}):

    vgg16 = tf.keras.applications.VGG16(input_tensor=inputs,weights='imagenet',include_top=False)
    f1 = vgg16.get_layer('block5_pool').output

    x = compose(
            DarknetConv2D(512, (1,1),name='block6_conv'),
            tf.keras.layers.BatchNormalization(name='block6_bn'),
            tf.keras.layers.LeakyReLU(alpha=0.1,name='block6_lrelu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block6_pool'))(f1)

    y_e = compose(
            DarknetConv2D(n_classes['emotion'], (1,1),name='block7_conv'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block7_pool'),
            tf.keras.layers.Flatten(name='block7_flat'))(x)

    y_g = compose(
            DarknetConv2D(128, (3,3),name='block8_conv1'),
            tf.keras.layers.BatchNormalization(name='block8_bn'),
            tf.keras.layers.LeakyReLU(alpha=0.1,name='block8_lrelu'),
            DarknetConv2D(n_classes['gender'], (1,1),name='block8_conv2'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block8_pool'),
            tf.keras.layers.Flatten(name='block8_flat'))(x)
    

    f2 = vgg16.get_layer('block5_conv3').output

    x = compose(
            DarknetConv2D(512, (1,1),name='block9_conv1'),
            tf.keras.layers.UpSampling2D(4,name='block9_upsample'))(x)
            
    x = tf.keras.layers.Concatenate(name='block9_concat')([x,f2])

    y_a = compose(
            DarknetConv2D(256, (1,1),name='block9_conv2'),
            tf.keras.layers.BatchNormalization(name='block9_bn'),
            tf.keras.layers.LeakyReLU(alpha=0.1,name='block9_lrelu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool1'),
            DarknetConv2D(256, (1,1),name='block9_conv3'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool2'),
            DarknetConv2D(n_classes['age'], (1,1),name='block9_conv4'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool3'),
            tf.keras.layers.Flatten(name='block9_flat'))(x)


    return tf.keras.models.Model(inputs = inputs, outputs=[y_e,y_g, y_a], name='ega_model')

def check_trainable_variables(model):
    trainable_vars = []
    static_vars = []
    for layer in model.layers:
        if layer.trainable:
            trainable_vars.append(layer.name)
        else:
            static_vars.append(layer.name)

    return {'trainable':trainable_vars, 'static': static_vars}
    
def set_non_trainable(model, block_ids = ['block9','block8']):
    for layer in model.layers:
        n_start = layer.name.split('_')[0]
        if  n_start in block_ids:
            layer.trainable = False
    return model

def compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a):
    loss_y_e = 0; loss_y_g = 0; loss_y_a = 0

    if target_y_e:
        l_y_e = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_y_e, logits=pred_y_e)
        loss_y_e = tf.reduce_mean(tf.reduce_sum(l_y_e,axis=[1]))

    if target_y_g:
        l_y_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_y_g, logits=pred_y_g)
        loss_y_g = tf.reduce_mean(tf.reduce_sum(l_y_g,axis=[1]))

    if target_y_a:
        l_y_a = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_y_a, logits=pred_y_a)
        loss_y_a = tf.reduce_mean(tf.reduce_sum(l_y_a,axis=[1]))
    
    total_loss = loss_y_a + loss_y_g + loss_y_e

    return total_loss

def train():
    image_input = Input(shape=(128,128, 3))

    model = create_model(image_input)
    model = set_non_trainable(model,block_ids=['block9','block8'])
    print(check_trainable_variables(model))

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    writer = tf.summary.create_file_writer("./log")
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)


    EPOCHS = 10
    STEPS = 4000
    batch_size = 2
    lambda_scale = 1.

    synthetic_dataset_path="./synthetic_dataset"
    TrainSet = DataGenerator(synthetic_dataset_path, batch_size)

    for epoch in range(EPOCHS):
        for step in range(STEPS):
            global_steps.assign_add(1)
            image_data, target_y_e, target_y_g, target_y_a = next(TrainSet)
            with tf.GradientTape() as tape:
                pred_y_e, pred_y_g, pred_y_a = model(image_data)
                loss = compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a )
                total_loss = loss
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print("=> epoch %d  step %d  total_loss: %.6f" %(epoch+1, step+1, total_loss.numpy()))
            # writing summary data
            with writer.as_default():
                tf.summary.scalar("total_loss", total_loss, step=global_steps)
                # tf.summary.scalar("score_loss", score_loss, step=global_steps)
                # tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
            writer.flush()
        model.save_weights("RPN.h5")





image_input = tf.keras.layers.Input(shape=(128,128, 3))

model = create_model(image_input)
model = set_non_trainable(model,block_ids=['block9','block8'])
print(check_trainable_variables(model))
print(len(check_trainable_variables(model)['trainable']))
print(len(model.trainable_variables))
print([x.name for x in model.trainable_variables])