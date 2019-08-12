from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import concatenate


seg_epochs = 25
classification_epochs = 10


def compact_cnn_segmentation(shape=(512,512,1),segmodelpath=None):
    epochs = seg_epochs
    visible = Input(shape=shape)
    # Block 1
    com1 = Conv2D(32,kernel_size=11,strides=2,activation='relu',kernel_initializer='random_normal',padding='same')(visible)
    bn1 = BatchNormalization()(com1)

    com2 = Conv2D(32,kernel_size=11,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn1)
    bn2 = BatchNormalization()(com2)

    com3 = Conv2D(32,kernel_size=11,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn2)
    bn3 = BatchNormalization()(com3)

    # Block 2
    com4 = Conv2D(64,kernel_size=7,strides=2,activation='relu',kernel_initializer='random_normal',padding='same')(bn3)
    bn4 = BatchNormalization()(com4)

    com5 = Conv2D(64,kernel_size=7,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn4)
    bn5 = BatchNormalization()(com5)

    com6 = Conv2D(64,kernel_size=7,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn5)
    bn6 = BatchNormalization()(com6)

    # Block 3
    com7 = Conv2D(128,kernel_size=3,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn6)
    bn7 = BatchNormalization()(com7)

    com8 = Conv2D(128,kernel_size=3,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn7)
    bn8 = BatchNormalization()(com8)

    com9 = Conv2D(128,kernel_size=3,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(bn8)
    bn9 = BatchNormalization(name='bn9')(com9)

    # Segmentation Layer
    seg1 = Conv2D(1,kernel_size=1,strides=1,activation='tanh',kernel_initializer='random_normal',padding='same')(bn9)
    bn10 = BatchNormalization(name='bn10')(seg1)
    
    segmodel = Model(inputs=visible,outputs=bn10)
    if segmodelpath != None:
        segmodel.load_weights(segmodelpath)
    segmodel.compile(optimizer='adadelta',loss='mean_squared_error')
#     print(segmodel.summary())
    return segmodel, epochs

def compact_cnn_classification(segmodelpath=None):
    if (segmodelpath == None):
        raise Exception('Specifiy segmentation model weights path')
    epochs = classification_epochs
    segmodel,_, = compact_cnn_segmentation(segmodelpath=segmodelpath)
    seg2 = GlobalMaxPooling2D()(segmodel.get_layer('bn10').output)
    bn11 = BatchNormalization()(seg2)

    seg3 = GlobalAveragePooling2D()(segmodel.get_layer('bn10').output)
    bn12 = BatchNormalization()(seg3)

    class1 = Conv2D(32,kernel_size=1,strides=1,activation='relu',kernel_initializer='random_normal',padding='same')(segmodel.get_layer('bn9').output)
    bn13 = BatchNormalization()(class1)

    class2 = GlobalMaxPooling2D()(bn13)
    bn14 = BatchNormalization()(class2)

    class3 = GlobalAveragePooling2D()(bn13)
    bn15 = BatchNormalization()(class3)

    poolmerge = concatenate([bn11,bn12,bn14,bn15])

    slayer = Dense(1,activation='sigmoid',kernel_initializer='random_normal',name='slayer')(poolmerge)
    classmodel = Model(inputs=segmodel.input,outputs=slayer)
    for i in range(21):
        classmodel.layers[i].trainable = False 
    classmodel.compile(optimizer='adadelta',loss='binary_crossentropy')#
    
#     print(classmodel.summary())
    return classmodel, epochs

def compact_cnn_model(segmodelpath=None,classmodelpath=None):
    if (segmodelpath == None):
        raise Exception('Specifiy segmentation model weights path')
    if (classmodelpath == None):
        raise Exception('Specifiy classification model weights path')
    modelinter,_,= compact_cnn_classification(segmodelpath=segmodelpath)
    modelinter.load_weights(classmodelpath)
    model = Model(inputs=modelinter.input,outputs=[modelinter.get_layer('bn10').output,modelinter.get_layer('slayer').output])
#     print(model.summary())
    return model