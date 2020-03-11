import numpy as np
import pandas as pd
import emotion_gender_age_model as ega

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


# split data into train, validation and test datasets in pandas
df = pd.read_pickle("./imfdb_meta.pkl")
train_test_mask = np.random.rand(len(df)) < 0.8
train_valid_df = df[train_test_mask]
test_df = df[~train_test_mask]

train_validation_mask = np.random.rand(len(train_valid_df)) < 0.9
train_df = train_valid_df[train_validation_mask]
valid_df = train_valid_df[~train_validation_mask]

batch_size = 12; image_size = 128
n_classes={'emotion':7,'age':101,'gender':2}
print(n_classes)
train_gen = ega.DataGenerator(train_df,batch_size=batch_size,image_size=image_size,augment=True, n_classes=n_classes )

# l = len(train_gen)

# x,yi,y2,y3 = train_gen[np.random.randint(l)]

# print(np.shape(x))
# print(np.mean(x))
# print(type(x[0,0,0,0]))

validation_gen = ega.DataGenerator(valid_df,batch_size=batch_size,image_size=image_size,augment=False, n_classes=n_classes )

model = ega.get_model(train_gen,non_trainable_blocks=['block9']) # 'bolck1','bolck2','bolck3','bolck4','bolck5',

fake_data = np.ones(shape=[12, image_size,image_size, 3]).astype(np.float32)
model(fake_data) # initialize model to load weights
model.load_weights("./checkpoints/run3/EGA_epoch_22_score_91.h5")

print(model.summary())
# print(train_gen.n_classes)
# print(train_gen.n_classes_check)
ega.train_model(model,train_gen, validation_gen,epochs=40,steps=5000,checkpoints_path="checkpoints/run4")