import os, sys, time, random, math
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import wraps, reduce
import Augmentor
from keras.utils import Sequence, to_categorical
import tensor_board_wrapper as tbw

tb = tbw.TensorBoardWrapper() 

physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 

def mk_dir(_dir):
    try:
        os.mkdir(_dir)
    except OSError:
        pass

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
    p.rotate(probability=0.6, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=0.6, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.6, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=0.6, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(input_image: Image):
        image = [input_image]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class DataGenerator(Sequence):
    def __init__(self, df: pd.DataFrame,  batch_size=32, image_size=128, n_classes={'emotion':6,'age':101,'gender':2}, augment=True):
        self.df = df
        self.augment = augment
        self.n_classes = n_classes
        self.n_classes_check = dict(n_classes)
        cols_needed = ['path','gender','emotion','age']
        for col in cols_needed:
            if col not in df.columns.values :
                self.n_classes_check[col] = False
                print("Expected to see "+ col + " but not found in the dataframe provided for face generator")
        
        
        self.image_num = self.df.shape[0]
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_shape = (image_size, image_size, 3)
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float)
        y_e = np.zeros((batch_size, 1), dtype=np.int32)
        y_g = np.zeros((batch_size, 1), dtype=np.int32)
        y_a = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            row_in_df = self.df.iloc[sample_id,:]
            try:
                image = Image.open(row_in_df.path)
                rescaled_image = rescale_image(image,input_shape=(self.image_size, self.image_size) )
                if self.augment:
                    processed_image = np.array(self.transform_image(rescaled_image))/255.
                else:
                    processed_image = np.array(rescaled_image,dtype=float)/255.
                x[i] = processed_image

                if self.n_classes_check['emotion']:
                    y_e[i] = row_in_df.emotion

                if self.n_classes_check['gender']:
                    y_g[i] = row_in_df.gender

                if self.n_classes_check['age']:
                    age = row_in_df.age
                    age += math.floor(np.random.randn() * 2 + 0.5)
                    y_a[i] = np.clip(age, 0, 100)
            except BaseException as e:
                # print(e)
                pass
        
        y_emotion = to_categorical(y_e, self.n_classes['emotion']) if self.n_classes_check['emotion'] else None
        y_gender = to_categorical(y_g, self.n_classes['gender']) if self.n_classes_check['gender'] else None
        y_age = to_categorical(y_a, self.n_classes['age']) if self.n_classes_check['age'] else None
        
        return x, y_emotion , y_gender, y_age
    
    def get_data(self):
        data_len = self.__len__()
        while True:
            yield self.__getitem__(np.random.randint(data_len))

    def get_sample_data(self, n=1):
        if n> self.__len__():
            n = self.__len__()
        return self.__getitem__(np.random.randint(n))


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
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(f1)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv5')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv6')(x)

    x = compose(
            DarknetConv2D(512, (1,1),name='block6_conv'),
            tf.keras.layers.BatchNormalization(name='block6_bn'),
            tf.keras.layers.LeakyReLU(alpha=0.1,name='block6_lrelu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block6_pool'))(x)

    y_e = compose(
            DarknetConv2D(256, (1,1),name='block7_conv'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block7_pool'),
            tf.keras.layers.Flatten(name='block7_flat'),
            tf.keras.layers.Dense(n_classes['emotion'],activation='softmax',name='block7_dense'))(x)

    y_g = compose(
            DarknetConv2D(128, (3,3),name='block8_conv1'),
            tf.keras.layers.BatchNormalization(name='block8_bn'),
            tf.keras.layers.LeakyReLU(alpha=0.1,name='block8_lrelu'),
            DarknetConv2D(64, (1,1),name='block8_conv2'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block8_pool'),
            tf.keras.layers.Flatten(name='block8_flat'),
            tf.keras.layers.Dense(n_classes['gender'],activation='softmax',name='block8_dense'))(x)
    

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
            DarknetConv2D(128, (3,3),name='block9_conv3'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same",name='block9_pool2'),
            tf.keras.layers.Flatten(name='block9_flat'),
            tf.keras.layers.Dense(n_classes['age'],activation='softmax',name='block9_dense'))(x)


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

    if target_y_e is not None:
        l_y_e = tf.keras.losses.categorical_crossentropy(target_y_e, pred_y_e)
        loss_y_e = tf.reduce_mean(l_y_e)

    if target_y_g is not None:
        l_y_g = tf.keras.losses.categorical_crossentropy(target_y_g, pred_y_g)
        loss_y_g = tf.reduce_mean(l_y_g)

    if target_y_a is not None:
        l_y_a = tf.keras.losses.categorical_crossentropy(target_y_a, pred_y_a)
        loss_y_a = tf.reduce_mean(l_y_a)

    return loss_y_e, loss_y_g, loss_y_a

def get_model(data_gen: DataGenerator,non_trainable_blocks=['block9','block8']):
    image_input = tf.keras.layers.Input(shape=data_gen.input_shape)

    model = create_model(image_input, n_classes=data_gen.n_classes)
    model = set_non_trainable(model,block_ids=non_trainable_blocks)
    return model

def train_model(model :tf.keras.models.Model , train_gen:DataGenerator, validation_gen:DataGenerator, epochs=10,steps=4000, checkpoints_path="checkpoints"):
    mk_dir(checkpoints_path)
    model_dir = os.path.join(checkpoints_path,tb.time_stamp)
    tf.saved_model.save(model,model_dir+"/model")
    optimizer = tf.keras.optimizers.Adadelta()
    
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    validation_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    tb.generate_model_graph(model,train_gen.get_sample_data())
    tb.launch_tensorboard()

    for epoch in range(epochs):
        for step in range(steps):
            global_steps.assign_add(1)
            image_data, target_y_e, target_y_g, target_y_a = next(train_gen.get_data())
            with tf.GradientTape() as tape:
                pred_y_e, pred_y_g, pred_y_a = model(image_data)
                l_e, l_g, l_a= compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a )
                total_loss = l_e + l_g + l_a
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print("=> epoch %d  step %d  train_loss: %.6f" %(epoch+1, step+1, total_loss.numpy()))
                tb.add_scalar("train/total_loss", total_loss, step=global_steps)
                tb.add_scalar("train/emotion_loss", l_e, step=global_steps)
                tb.add_scalar("train/gender_loss", l_g, step=global_steps)
                tb.add_scalar("train/age_loss", l_a, step=global_steps)
            

            # validation step
            if step%500==0:
                validation_steps.assign_add(1)
                image_data, target_y_e, target_y_g, target_y_a = next(validation_gen.get_data())
                pred_y_e, pred_y_g, pred_y_a = model(image_data)
                l_e, l_g, l_a = compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a )
                total_valid_loss = l_e + l_g + l_a
                tb.add_scalar("valid_loss", total_valid_loss, step=validation_steps)

        p_loss = int(round(total_loss.numpy(),2)*100)
        model.save(f"{model_dir}/EGA_epoch_{epoch}_score_{p_loss}.model")
        model.save_weights(f"{model_dir}/EGA_epoch_{epoch}_score_{p_loss}.h5")

def load_model(path):
    model = tf.keras.models.load_model(path)
    print(model.summary())
    return model

def to_numpy(x):
    print(type(x))
    if type(x) == tf.Tensor:
        return x.numpy()
    else:
        return x


import mlflow
import mlflow.tensorflow
import mlflow.keras
import tempfile




def train_model_mlflow(model :tf.keras.models.Model , train_gen:DataGenerator, validation_gen:DataGenerator, epochs=10,steps=4000,mlflow_server='http://0.0.0.0:8643',checkpoints_path="checkpoints/run3"):
    # Configure output_dir
    output_dir = tempfile.mkdtemp()
    
    if mlflow_server:
        # Tracking URI
        if not mlflow_server.startswith("http"):
            mlflow_tracking_uri = 'http://' + mlflow_server + ':5000'
        else:
            mlflow_tracking_uri = mlflow_server
        # Set the Tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print("MLflow Tracking URI: %s" % mlflow_tracking_uri)
    else:
        print("MLflow Tracking URI: %s" % "local directory 'mlruns'")
    
    # mlflow.tensorflow.autolog()
    # mlflow.keras.autolog()
    mlflow.set_experiment("/face-age-emotion-gender-detector")
    
    with mlflow.start_run():
        model_dir = "models/"+ str(mlflow.active_run().info.run_uuid)
        # mlflow.log_artifacts("checkpoints/")
        mlflow.log_param('Epochs',str(epochs))
        mlflow.log_param('Steps',str(steps))
        x = str(model.summary())
        mlflow.log_param('model',x )
        mlflow.keras.log_model(model,'models')
        tf.saved_model.save(model,model_dir)
        mlflow.log_artifact('./' + model_dir+"/saved_model.pb")
        mlflow.log_artifacts('./'+model_dir,artifact_path='models')

        optimizer = tf.keras.optimizers.Adadelta()
        global_steps = 0
        validation_steps = 0
        for epoch in range(epochs):
            for step in range(steps):
                global_steps += 1
                image_data, target_y_e, target_y_g, target_y_a = next(train_gen.get_data())
                with tf.GradientTape() as tape:
                    pred_y_e, pred_y_g, pred_y_a = model(image_data)
                    l_e, l_g, l_a= compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a )
                    total_loss = l_e + l_g + l_a
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    print("=> epoch %d  step %d  train_loss: %.6f" %(epoch+1, step+1, total_loss.numpy()))

                    mlflow.log_metric("train/total_loss", total_loss.numpy(), step=global_steps)
                    mlflow.log_metric("train/emotion_loss", l_e.numpy(), step=global_steps)
                    mlflow.log_metric("train/gender_loss", l_g.numpy(), step=global_steps)
                    mlflow.log_metric("train/age_loss", l_a, step=global_steps)

                # validation step
                if step%500==0:
                    validation_steps += 1
                    image_data, target_y_e, target_y_g, target_y_a = next(validation_gen.get_data())
                    pred_y_e, pred_y_g, pred_y_a = model(image_data)
                    l_e, l_g, l_a = compute_loss(pred_y_e, pred_y_g, pred_y_a, target_y_e, target_y_g, target_y_a )
                    total_valid_loss = l_e + l_g + l_a
                    mlflow.log_metric("valid_loss", total_valid_loss.numpy(), step=validation_steps)
   

            mk_dir("checkpoints/")
            p_loss = int(round(total_loss.numpy(),2)*100)
            print(f"EGA_epoch_{epoch}_score_{p_loss}")
            # model.save(f"EGA_epoch_{epoch}_score_{p_loss}")
            # model.save_weights(f"EGA_epoch_{epoch}_score_{p_loss}.h5")

            mlflow.keras.save_model(model,"checkpoints/"+str(int(time.time())))

        mlflow.log_artifacts("checkpoints/")
        mlflow.end_run()




























if __name__=='__main__':
    image_input = tf.keras.layers.Input(shape=(128,128, 3))

    model = create_model(image_input, n_classes={'emotion':6,'age':101,'gender':2})
    model = set_non_trainable(model,block_ids=['block9','block8'])
    # print(check_trainable_variables(model))
    # print(len(check_trainable_variables(model)['trainable']))
    # print(len(model.trainable_variables))
    # print([x.name for x in model.trainable_variables])

    test_input = np.random.rand(2,128,128,3)
    pred_y_e, pred_y_g, pred_y_a = model(test_input)
    print(np.shape(pred_y_a), type(pred_y_a))

    target_y_a = to_categorical(np.random.randint(101,size=(2,1)),num_classes=101)
    loss_y_a, loss_y_g, loss_y_e = compute_loss(pred_y_e, pred_y_g, pred_y_a, None, None, target_y_a )

    print(loss_y_a)
    

    # print(model.summary())