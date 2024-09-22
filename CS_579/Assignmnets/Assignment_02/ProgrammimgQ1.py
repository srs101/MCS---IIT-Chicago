#!/usr/bin/env python
# coding: utf-8

# # Sahil Sheikh
# ## CWID: A20518693
# ## Subject:CS 577
# ## Semester: FALL 22
# ## ASSIGNMENT 2
# Programming part 1

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_iris
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint


# In[2]:


def load_dataset():
    #loading the data from site
    #datacsv = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    #datacsv.head()
    # we tried using the above method but then it's difficult to one hot encode the de to some invalid literal for int() with base 10: 'Iris-setosa' error
    load_data = load_iris()
    # seperating the data and labels
    dataX = load_data['data']
    labelY = load_data['target']
    names = load_data['target_names']
    feature_names = load_data['feature_names']
    
    #one hot encoding the last column containing the classes
    hotenc = OneHotEncoder()
    label = hotenc.fit_transform(labelY[:, np.newaxis]).toarray()

    
    # Normalizing the data
    mean = dataX.mean(axis=0)
    dataX -= mean
    std = dataX.std()
    dataX /= std
    data = dataX


    #spliting the dataset into training,testing and validation set 
    train, test, train_lab, test_lab = train_test_split(data,label,test_size=0.1, random_state=3)
    train_split, vali, train_lab_sp, vali_lab = train_test_split(train,train_lab,test_size=0.1)
    train = train_split
    train_lab = train_lab_sp
    print("train,test shape")
    print(train.shape)
    print(test.shape)
    print("train target ,test shape") 
    print(train_lab.shape)
    print(test_lab.shape)    

    
    #visualizing the dataset
    plt.figure(figsize=(16, 6))
    for target, target_name in enumerate(names):
        X_plot = dataX[labelY == target]
        plt.plot(X_plot[:, 0], X_plot[:, 1], linestyle='none', marker='o', label=target_name)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.axis('equal')
    plt.legend();
    return train, test, train_lab, test_lab,vali,vali_lab


# In[3]:


train, test, train_lab, test_lab,vali,vali_lab = load_dataset()


# In[4]:


#hyperparameters
epochs = 5
batch_size = 20
loss_func  = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
opt = tf.keras.optimizers.Adam(learning_rate=0.09)
eval_metric = ['accuracy']
num_labels= 3
#metrics
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


# In[5]:


def model(act_func,opt,loss_func):
    model = tf.keras.Sequential([
        keras.layers.Dense(3, activation='sigmoid', input_dim=4),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(3, activation=act_func)
    ])
    model.compile(loss=loss_func,optimizer=opt,metrics='accuracy')
    return model


# In[6]:


model_1 =model('softmax',opt,loss_func)
model_1.summary()


# In[7]:


# Preparing the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train, train_lab))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Preparing the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((vali, vali_lab))
val_dataset = val_dataset.batch(batch_size)


# In[8]:


from keras.callbacks import CallbackList
cp_callback = ModelCheckpoint(filepath='model_1weights.h5',
                               monitor='val_loss',
                               save_weights_only=True,
                               save_best_only=True,
                               mode='auto',
                               save_freq='epoch',
                               verbose=1)
callbacks = CallbackList(cp_callback, add_history=True, model=model_1)
logs = {}
callbacks.on_train_begin(logs=logs)


# In[9]:


#training loop
def train_loop(model):
    
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]

    for epoch in range(epochs):
        print("\nEpoch no %d" %(epoch,))
        for step,(x_batch_train, y_batch_train) in enumerate(train_dataset):


            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_val = loss_func(y_batch_train, logits)
            grads = tape.gradient(loss_val,model.trainable_weights)
            opt.apply_gradients(zip(grads,model.trainable_weights))  
            train_acc_metric.update_state(y_batch_train, logits)
            logs["train_loss"] = loss_val
            callbacks.on_train_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)
        print("Training loss for epoch %d : %.6f"% (epoch, float(loss_val)))
        train_loss_list.append(float(loss_val))
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.6f" % (float(train_acc),))
        train_acc_list.append(train_acc)
        train_acc_metric.reset_states()

        callbacks.on_batch_begin(step, logs=logs)
        callbacks.on_test_batch_begin(step, logs=logs)

        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_func(y_batch_val, val_logits)
            val_acc_metric.update_state(y_batch_val, val_logits)
            logs["val_loss"] = val_loss_value
            callbacks.on_test_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)
        print("Validation loss for epoch %d : %.6f"% (epoch, float(val_loss_value)))
        val_loss_list.append(float(val_loss_value))
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.6f" % (float(val_acc),))
        val_acc_list.append(float(val_acc))  
        callbacks.on_epoch_end(epoch, logs=logs)

    callbacks.on_train_end(logs=logs)

    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()
    print('\n\nAccuracy of model is ', "{:.2f}".format((max(train_acc_list))*100),'%') 
    return 0


# In[10]:


train_loop(model_1)


# # Best model

# In[11]:


def modelv2(act_func,opt,loss_func):
    model = tf.keras.Sequential([
        keras.layers.Dense(4, activation='sigmoid', input_dim=4),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(3, activation=act_func)
    ])
    model.compile(loss=loss_func,optimizer=opt,metrics='accuracy')
    return model


# In[12]:


model_2 = modelv2('softmax',opt,loss_func)
train_loop(model_2)


# In[13]:


model = modelv2('softmax',opt,loss_func)
tuned_callback = ModelCheckpoint(filepath='best_weights.h5',
                               monitor='val_loss',
                               save_weights_only=True,
                               save_best_only=True,
                               mode='auto',
                               save_freq='epoch',
                               verbose=1)
callbacks = CallbackList(tuned_callback, add_history=True, model=model)
logs = {}
callbacks.on_train_begin(logs=logs)

train_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]

for epoch in range(epochs):
    print("\nEpoch no %d" %(epoch,))
    for step,(x_batch_train, y_batch_train) in enumerate(train_dataset):


        callbacks.on_batch_begin(step, logs=logs)
        callbacks.on_train_batch_begin(step, logs=logs)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_val = loss_func(y_batch_train, logits)
        grads = tape.gradient(loss_val,model.trainable_weights)
        opt.apply_gradients(zip(grads,model.trainable_weights))  
        train_acc_metric.update_state(y_batch_train, logits)
        logs["train_loss"] = loss_val
        callbacks.on_train_batch_end(step, logs=logs)
        callbacks.on_batch_end(step, logs=logs)
    print("Training loss for epoch %d : %.6f"% (epoch, float(loss_val)))
    train_loss_list.append(float(loss_val))
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.6f" % (float(train_acc),))
    train_acc_list.append(train_acc)
    train_acc_metric.reset_states()

    callbacks.on_batch_begin(step, logs=logs)
    callbacks.on_test_batch_begin(step, logs=logs)

    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_loss_value = loss_func(y_batch_val, val_logits)
        val_acc_metric.update_state(y_batch_val, val_logits)
        logs["val_loss"] = val_loss_value
        callbacks.on_test_batch_end(step, logs=logs)
        callbacks.on_batch_end(step, logs=logs)
    print("Validation loss for epoch %d : %.6f"% (epoch, float(val_loss_value)))
    val_loss_list.append(float(val_loss_value))
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.6f" % (float(val_acc),))
    val_acc_list.append(float(val_acc))  
    callbacks.on_epoch_end(epoch, logs=logs)

callbacks.on_train_end(logs=logs)

plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()
print('\n\nAccuracy of model is ', "{:.2f}".format((max(train_acc_list))*100),'%') 


# In[17]:


def evaluation(filename):
    model = modelv2('softmax',opt,loss_func)
    model.load_weights(filename)
  # evaluate model
    test_acc = model.evaluate(test,test_lab, verbose=1)
    return test_acc


# In[18]:


test_acc = evaluation('best_weights.h5')
print("Test accuracy of model is ",test_acc) 


# In[ ]:




