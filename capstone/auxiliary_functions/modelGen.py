import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras import optimizers
from keras.optimizers import RMSprop


def create_benchmark_CNN(plantsImageSize = 224, drop_outA = 0.1, drop_outB = 0.1, neurons = 500):
    '''
    Simple Sequential CNN that will be used as a benchmark
    '''
    benchmark = Sequential()

    # CNN model architecture
    benchmark.add(Conv2D(filters=16, kernel_size=2, strides = 1, padding='same', activation = 'relu',
                         input_shape=(plantsImageSize,plantsImageSize,3)))
    benchmark.add(Conv2D(filters=16, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(MaxPooling2D(pool_size=2))

    benchmark.add(Conv2D(filters=32, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(Conv2D(filters=32, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(Dropout(drop_outA))
    benchmark.add(MaxPooling2D(pool_size=(2,2)))

    benchmark.add(Conv2D(filters=64, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(Conv2D(filters=64, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(Dropout(drop_outB))
    benchmark.add(MaxPooling2D(pool_size=(2,2)))

    benchmark.add(Flatten())
    benchmark.add(Dense(neurons, activation='relu'))
    benchmark.add(Dense(12, activation='softmax'))     # There are 12 Plant Seedling classes

    # Compile the model
    benchmark.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return benchmark



def load_pretrain_CNN(modelName = 'VGG19', imageSize = 224, layers_to_keep = -1):
    ''' 
    Loads and returns VGG16, VGG19, ResNet50, or InceptionV3 model with last several layers 
    removed. Additionally layers may be removed by increasing input num_layers_to_remove.
    '''
    
    if (modelName == 'VGG16'):
        model = VGG16(include_top=False, weights='imagenet', input_shape = (imageSize,imageSize,3))
    elif (modelName == 'VGG19'):
        model = VGG19(include_top=False, weights='imagenet', input_shape = (imageSize,imageSize,3))
    elif (modelName == 'ResNet50'):
        model = ResNet50(include_top=False, weights='imagenet', input_shape = (imageSize,imageSize,3))
    elif (modelName == 'InceptionV3'):
        model = InceptionV3(include_top=False, weights='imagenet', input_shape = (imageSize,imageSize,3))
    else:
        raise ValueError('Invalid model input')  
        
    # Remove layers
    if (layers_to_keep != -1):
        num_layers_to_remove = len(model.layers) - layers_to_keep
        for j in range(0,num_layers_to_remove):
            model.layers.pop()
            #copiedModel.outputs = [copiedModel.layers[-1].output]
            #copiedModel.output_layers = [copiedModel.layers[-1]] 
            #copiedModel.layers[-1].outbound_nodes = []
    return model



def generate_bottleneck(thisModel, data_files, inputImageSize = 224):
    ''' Passes data through a pre-trained model in order to generate bottleneck features '''
    
    # Data
    preprocessedData = preprocess_input(paths_to_tensor(data_files, inputImageSize))
    bottleneck_features = thisModel.predict(preprocessedData)
    return bottleneck_features, preprocessedData


def create_top_model(inputData, learningRate = 0.001, numDenseNeurons = 256, dropout_rate = 0.2):
    ''' 
    A small, densely connected neural network to place on top of another CNN being used for transfer learning.
    Borrowed from:
        https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    '''
    top_model = Sequential()
    top_model.add(Flatten(input_shape = inputData.shape[1:]))
    top_model.add(Dense(numDenseNeurons, activation='relu'))
    top_model.add(Dropout(dropout_rate))
    top_model.add(Dense(12, activation='softmax'))
    rmsprop = RMSprop(lr=learningRate, rho=0.9, epsilon=1e-08, decay=0.0)
    top_model.compile(loss='categorical_crossentropy', optimizer = rmsprop, metrics=['accuracy'])
    return top_model


def uncompiled_top_model(baseModel, numDenseNeurons, dropout_rate):
    top_model = Sequential()
    top_model.add(Flatten(input_shape = baseModel.output_shape[1:]))
    top_model.add(Dense(numDenseNeurons, activation='relu'))
    top_model.add(Dropout(dropout_rate))
    top_model.add(Dense(12, activation='softmax'))
    return top_model


def aggregated_model(plantsImageSize,vggLayerstoKeep,nLayersToFreeze,learnRate,numDenseNeurons,dropout_rate,optim):
    ''' Returns a compiled deep convolutional neural network. It consists of layers of the VGG19 network,
        trained on ImageNet, with a small densely connected "top network" added to it. The user specifies the 
        number of VGG19 layers to keep, the number of VGG19 layers to freeze, number of neurons for the densely connected
        top network, the choice between RMSprop or SGD optimizer, and learning rate if the RMSprop optimizer is chosen'''
    
    vgg = VGG19(include_top=False, weights='imagenet', input_shape = (plantsImageSize,plantsImageSize,3))

    # Copy the VGG model by brute force
    model = Sequential()
    counter = 0
    for layer in vgg.layers:
        if counter < vggLayerstoKeep:
            model.add(layer)
            counter = counter + 1

    # Set the weights just to be safe
    model.set_weights(vgg.get_weights())

    # Freeze certain number of VGG19 layers
    for layer in model.layers[0:nLayersToFreeze]:
        layer.trainable = False

    # Add a pooling layer if necessary
    if (len(model.layers) not in [22, 21, 16, 17, 11, 12, 6, 7, 3, 4]):
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    # Explicitly add layers of a topmodel to the copied VGG layers
    shapeForFlattenLayer = model.output_shape[1:]
    model.add(Flatten(input_shape = shapeForFlattenLayer))
    model.add(Dense(numDenseNeurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(12, activation='softmax'))   # 12 is for the number of plant classes

    # Compile the model with one of two possible optimizers
    if optim == 0:
        rmsprop = RMSprop(lr=learnRate, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer = rmsprop, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', 
                              optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), 
                              metrics=['accuracy'])
    return model


# The number of VGG19 layers to freeze depends on the total number of layers in the model.
# This function freezes all layers up to the last block of convolutional layers, which
# are not frozen, and thus are allowed to be trained.

def num_transfer_layers_to_freeze(num_layers_to_keep):
    if num_layers_to_keep == -1 or num_layers_to_keep == 22:
        numLayersToFreeze = 17
    elif (num_layers_to_keep <= 17 and num_layers_to_keep > 12):
        numLayersToFreeze = 12
    elif (num_layers_to_keep <= 12 and num_layers_to_keep > 7):
        numLayersToFreeze = 7
    elif (num_layers_to_keep <= 7 and num_layers_to_keep > 4):
        numLayersToFreeze = 4
    elif num_layers_to_keep == 4:
        numLayersToFreeze = 1
    return numLayersToFreeze