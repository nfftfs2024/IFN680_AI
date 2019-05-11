# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:10:44 2018
Project Team:
    n10143416 Dmitrii Menshikov
    n10030581 Chen-Yen Chou
    n10080236 Victor Manuel Villamil
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
import random
from collections import Counter
from sklearn.metrics import accuracy_score

def calc_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

def contrastive_lost(y_true,y_pred):
    '''
    Contrastive loss function
    '''
    margin = 1
    return K.mean((1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vectors):
    '''
    Euclidean distance function. Measures distance between two vectors.
    @params:
        tuple (vector1,vector2)
    '''
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    '''
    returns shape for output of eucledean distance layer
    '''
    shape1, shape2 = shapes
    return (shape1[0], 1)

    
def MNISTsubset(dataset,numbers,qty=-1):    
    '''
    returns subset of MNIST dataset, where each of the subset is in numbers
    returns two np.ndarray : images and labels
    @params:
        dataset : MNIST dataset
        numbers : array or set of digits
        qty     : quantity for sampling
    '''
    (x_train, y_train), (x_test, y_test) = dataset
    y=np.concatenate((y_train,y_test))
    x=np.vstack((x_train,x_test))
    img_cols,img_rows=x[0].shape
    x = x.astype('float32')
    x /= 255
    x=x.reshape(x.shape[0], img_rows, img_cols, 1)
    
    indexes=[t in numbers for t in y]
    images = x[indexes]
    labels = y[indexes]
    if qty==-1: #returns all available datapoints
        sample_ind=list(range(len(labels)))
    else:
        sample_ind=random.sample(list(range(len(labels))),qty)
    return images[sample_ind],labels[sample_ind]
#------------------------------------------------------------------------------
def create_baseline_cnn(input_shape, Conv_layers, dropout_rate, dense1,dense2,activation1,activation2):
    '''
    Creates sequential baseline network.
    @params:
        input_shape : input shape
        Conv_layers : quantity of convolution stacks (1 stack = Conv+Conv+MaxPool+Dropout)
        dropout_rate: Dropout rate foe all dropout layers
        dense1      : size of 1st Dense layer
        dense2      : size of 2nd Dense layer
        activation1 : activation function of 1st Dense layer
        activation2 : activation function of 2nd Dense layer
        
    '''
    baseline_cnn=keras.models.Sequential()
    for i in range(Conv_layers):
        baseline_cnn.add(keras.layers.Conv2D(32,activation='relu',kernel_size=(3,3),padding='same',input_shape=input_shape,))
        baseline_cnn.add(keras.layers.Conv2D(64,activation='relu',kernel_size=(3,3),padding='same'))
        baseline_cnn.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        baseline_cnn.add(keras.layers.Dropout(rate=dropout_rate))
    
    baseline_cnn.add(keras.layers.Flatten())
    #cnn_network.add(keras.layers.Softmax())
    baseline_cnn.add(keras.layers.Dense(dense1,activation=activation1))
    baseline_cnn.add(keras.layers.Dense(dense2,activation=activation2))
    return baseline_cnn
#------------------------------------------------------------------------------    
def create_siamese_model(baseline):
    '''
    Creates Siamese network, based on baseline network.
    @params:
        baseline : baseline network
    '''
    input_a =  keras.layers.Input(shape=baseline.layers[0].input_shape[1:])
    input_b = keras.layers.Input(shape=baseline.layers[0].input_shape[1:])  

    encoded_a=baseline(input_a)
    encoded_b=baseline(input_b)
    
    distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_a,encoded_b])   
    siam_model=keras.models.Model(inputs=[input_a,input_b],outputs=[distance])
    siam_model.compile(optimizer='adam',
                       #loss='mean_squared_error',
                       loss=contrastive_lost,
                       #loss=customLoss,
                       metrics=[calc_accuracy])
                       #metrics=['mse'])
    return siam_model
#------------------------------------------------------------------------------
def create_pairs_balanced(subset_1,subset_2,balanced=True):
    '''
    Creates pairs from two subsets of MNIST.
    @params:
        subset1    :  subset of MNIST. tuple (digits,labels)
        subset2    :  subset of MNIST. tuple (digits,labels)
        balanced   :  True for balanced dataset positives=negatives
    '''
    images1,labels1=subset_1
    images2,labels2=subset_2
    pairs=[]
    labels=[]
    description=[]
    pos=0
    neg=0
    if not balanced: pos=np.inf
    for i,image1 in enumerate(images1):
        for j,image2 in enumerate(images2):
            pair_description=str(labels1[i])+'X'+str(labels2[j])
            if labels1[i]==labels2[j]:
                label=0
                pos+=1
                labels.append(label)
                pairs.append([image1,image2])
                description.append(pair_description)
            elif pos>neg:
                label=1
                neg+=1
                labels.append(label)
                pairs.append([image1,image2])
                description.append(pair_description)
    
    return np.array(pairs),np.array(labels),description
    
#------------------------------------------------------------------------------ 
def split_mnist_dataset(dataset,val_size):
    '''
    Splits MNIST Subset into train and validation.
        @params:
            dataset - MNIST Subset (data,labels)
            val_size - 0 < validation size <1
    '''
    data,labels=dataset
    datalen=len(labels)
    all_indexes=np.arange(datalen)
    val_indexes = np.random.permutation(all_indexes)[:round(datalen*val_size)]
    train_indexes = list(set(all_indexes)-set(val_indexes))
    validation_dataset=(data[val_indexes],labels[val_indexes])
    train_dataset = (data[train_indexes],labels[train_indexes])
    return train_dataset, validation_dataset
#------------------------------------------------------------------------------
        
def create_and_train_model(train_volume,epochs,*params):
    '''
    Creates and trains Siamese Model.
    @params:
        train_volume     : volume of the train dataset 
        epochs           : number of Epochs
        params           : tuple of parameters for neural network 
    '''
    val_size=0.2
    
    report=''
    dataset=keras.datasets.mnist.load_data()
    train_dataset=MNISTsubset(dataset,[2,3,4,5,6,7],train_volume)
    train_dataset,validation_dataset = split_mnist_dataset(train_dataset,val_size)
    

    pairs,labels,d=create_pairs_balanced(train_dataset,train_dataset,True)
    val_pairs,val_labels,d = create_pairs_balanced(validation_dataset,validation_dataset,True)
    
    img_rows, img_cols = pairs.shape[2:4]
    input_shape = (img_rows, img_cols, 1)
    
    baseline_cnn=create_baseline_cnn(input_shape,*params)
    siam_model=create_siamese_model(baseline_cnn)
    #siam_model.summary()
    history=siam_model.fit([pairs[:,0],pairs[:,1]],
                   labels,
                   epochs=epochs,
                   #validation_split=val_size,
                   validation_data=([val_pairs[:,0],val_pairs[:,1]],val_labels),
                   batch_size=128)
    report+=", ".join([str(param) for param in params])
    print(report)
    return siam_model,history
#------------------------------------------------------------------------------
def evaluate_model(siam_model,dataset,test_size):    
    '''
    Evaluates model.
    @params:
        siam_model    : model for evaluation
        dataset       : MNIST dataset
        test_size     : volume of test sample
    '''
    threshold=0.5
    report=''
    
    train_dataset=MNISTsubset(dataset,[2,3,4,5,6,7],test_size)
    test_dataset=MNISTsubset(dataset,[0,1,8,9],test_size)
    union_dataset = MNISTsubset(dataset,[0,1,2,3,4,5,6,7,8,9],test_size)

    #report.append("prediction values: ")
    #report.append(str(siam_model.evaluate([pairs[:,0],pairs[:,1]],labels)))
    
    #report.append("accuracy train X train")
    pairs,labels,d = create_pairs_balanced(train_dataset,train_dataset,True)
    preds=siam_model.predict([pairs[:,0],pairs[:,1]])
    #threshold=find_threshold(labels,preds)
    report+=', '+str(accuracy_score(preds>threshold,labels))
    report+=' ,t'+str(threshold)
    
    #report.append("accuracy test X test")
    pairs,labels,d = create_pairs_balanced(test_dataset,test_dataset,True)
    preds=siam_model.predict([pairs[:,0],pairs[:,1]])
    #threshold=find_threshold(labels,preds)
    report+=', '+str(accuracy_score(preds>threshold,labels))
    report+=' ,t'+str(threshold)
    
    #report.append("accuracy test X train")
    pairs,labels,d = create_pairs_balanced(union_dataset,union_dataset,True)
    preds=siam_model.predict([pairs[:,0],pairs[:,1]])
    #threshold=find_threshold(labels,preds)
    report+=', '+str(accuracy_score(preds>threshold,labels))
    report+=' ,t'+str(threshold)
    #report.append("tran*train,test*test,train*test")

    return report
#------------------------------------------------------------------------------
def log_result(result):
    '''
    writes string to the end of file
    @params:
        result : string
    '''
    filename="rep.txt"
    f = open(filename, 'a')
    f.write(result)
    f.close() 
#------------------------------------------------------------------------------
def p_p(pair):
    '''
    Plots pair
    @params:
        pair - pair of images with shape (2,28,28,1)
    '''
    fig=plt.figure(figsize=(2, 2))
    fig.add_subplot(1,2,1)
    plt.imshow(pair[0].reshape(28,28))
    
    fig.add_subplot(1, 2,2)
    plt.imshow(pair[1].reshape(28,28))
    plt.show()
#------------------------------------------------------------------------------            
def perform_greed_search():
    '''
    performs greed search among the list of hyperparameters and prints results
    to the output file
    '''    
    result=[]
    neurons1=[512,1024]
    neurons2=[512,1024]
    activations1=['relu','sigmoid']
    activations2=['sigmoid','sigmoid']
    Conv_layers=[2,3]
    dropouts=[0.05,0.1,0.15,0.2]
    label="500/200"
    
    
    #Creating set of parameter for greed search
    params=[]
    for d1 in neurons1:
        for d2 in neurons2:
            for a1 in activations1:
                for a2 in activations2:
                    for convs in Conv_layers:
                        for drop in dropouts:
                            p=[convs,drop,d1,d2,a1,a2]
                            params.append(p)
                    
    #greed search
    i=0       
    dataset=keras.datasets.mnist.load_data()      
    for p in params:       
        print (str(i)+" / " + str(len(params)))
        i+=1
        print (p)
        siam_model=create_and_train_model(500,5,*p)
        result=label + ", ".join([str(param) for param in p])
        result+=evaluate_model(siam_model,dataset,200)
        print(result)
        log_result(result+'\n')
        # one more time
        result=label +", ".join([str(param) for param in p])
        result+=evaluate_model(siam_model,dataset,200)
        print(result)
        log_result(result+'\n')
        
        print("+-"*10)
#------------------------------------------------------------------------------
def visualize(siam_model,digits):
    '''
    Visualisation of TP,TN,FP and FN Pairs
    @params:
        Siam_Model - model 
        digits = tuple of digits to visualise
    '''
    subset=MNISTsubset(dataset,digits,100)
    pairs=create_pairs_balanced(subset,subset,True)

    pairs, labels, description = pairs
    prediction=siam_model.predict([pairs[:,0],pairs[:,1]])

    FP=[i for i in range(len(pairs)) if labels[i]==1 and prediction[i]<0.5 ]
    TP=[i for i in range(len(pairs)) if labels[i]==0 and prediction[i]<0.5 ]
    TN=[i for i in range(len(pairs)) if labels[i]==1 and prediction[i]>0.5 ]
    FN=[i for i in range(len(pairs)) if labels[i]==0 and prediction[i]>0.5 ]
    
    examples = (TP,TN,FP,FN)
    for t in examples:
        samp=random.sample(t,10)
        print("="*30)
        for i in samp:
            print( description[i] +", label: "+str(labels[i])+",distance: "+str(prediction[i]) )
            p_p(pairs[i])
              
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------
        
perform_greed_search()

siam_model,history=create_and_train_model(500,5,*(3,0.1,1024,1024,'relu','sigmoid'))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['calc_accuracy'])
plt.plot(history.history['val_calc_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


dataset=keras.datasets.mnist.load_data()  

#Evaluation
for i in range(5):
    print(evaluate_model(siam_model,dataset,100))

#visualisation
print("="*10+"Train"+"="*10)
visualize(siam_model,[2,3,4,5,6,7])
print("="*10+"Test"+"="*10)
visualize(siam_model,[0,1,8,9])
print("="*10+"Union"+"="*10)
visualize(siam_model,[0,1,2,3,4,5,6,7,8,9])