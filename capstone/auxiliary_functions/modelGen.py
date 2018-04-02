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



def create_benchmark_CNN(drop_out = [[0.1], [0.2]], neurons = 500):
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
    benchmark.add(Dropout(drop_out[0][0]))
#    benchmark.add(Dropout(0.10))
    benchmark.add(MaxPooling2D(pool_size=2))

    benchmark.add(Conv2D(filters=64, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
    benchmark.add(Conv2D(filters=64, kernel_size=2, strides = 1, padding='same', activation = 'relu'))
#    benchmark.add(Dropout(0.20))
    benchmark.add(Dropout(drop_out[1][0]))
    benchmark.add(MaxPooling2D(pool_size=2))

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
    return model


def generate_bottleneck_features(model, x_train, y_train, x_valid, y_valid, imageSize = 224, batchSize = 32):
    ''' Largely borrowed and adapted from 
        https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
    '''
    SEED = 0
    ROTATION = 180      # randomly rotate images up to 180 degrees
    WIDTH_SHIFT = 0.1   # randomly shift images horizontally (10% of total width)
    HEIGHT_SHIFT = 0.1  # randomly shift images vertically (10% of total height)
    SHEAR_RANGE = 0.2   # randomly shear images counter-clockwise in radians (0.2 rads ~ 11 degrees)
    
    numTrainSamples = len(y_train)
    numValidSamples = len(y_valid)

    # Image Augmentation for train set
    datagen_train = ImageDataGenerator(rotation_range = ROTATION,
                                       width_shift_range = WIDTH_SHIFT,  
                                       height_shift_range = HEIGHT_SHIFT,  
                                       shear_range = SHEAR_RANGE,
                                       fill_mode = 'nearest',
                                       horizontal_flip = True) # randomly flip images horizontally
    
    generator = datagen_train.flow(x_train, 
                                   y_train,
                                   target_size = (imageSize, imageSize),
                                   batch_size = batchSize,
                                   class_mode = None,
                                   shuffle = False,
                                   seed = SEED)
    
    bottleneck_features_train = model.predict_generator(generator, numTrainSamples // batchSize)
    
#    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    # Image Augmentation for Validation Set
    datagen_valid = ImageDataGenerator(rescale = 1.0 / imageSize)
    
    generator = datagen_valid.flow(x_valid, 
                                   y_valid,
                                   target_size = (imageSize, imageSize),
                                   batch_size = batchSize,
                                   class_mode = None,
                                   shuffle = False,
                                   seed = SEED)
    
    bottleneck_features_validation = model.predict_generator(generator, numValidSamples // batchSize)

#    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    return bottleneck_features_train, bottleneck_features_validation


#def generate_bottleneck(thisModel, inputImageSize = 224, train_files, valid_files, test_files):
#    ''' '''    
    # Training Data
#    train_input = preprocess_input(paths_to_tensor(train_files, inputImageSize))
#    bottleneck_train = thisModel.predict(train_input)
   
    # Validation Data
#    valid_input = preprocess_input(paths_to_tensor(valid_files, inputImageSize))
#    bottleneck_valid = thisModel.predict(valid_input)

    # Test Data
#    test_input = preprocess_input(paths_to_tensor(train_files, inputImageSize))
#    bottleneck_test = thisModel.predict(test_input)
    
#    return bottleneck_train, bottleneck_valid, bottleneck_test


def fully_connected_toplayer(InputShape, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999):
    ''' '''
    fullyConnected = Sequential()
    fullyConnected.add(GlobalAveragePooling2D(input_shape = InputShape))
    fullyConnected.add(Dense(12, activation='softmax'))
    
    # Compile the top model
    adam = optimizers.Adam(lr = learningRate, beta_1 = beta1, beta_2 = beta2, epsilon=None, decay=0.0, amsgrad=False)
    fullyConnected.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return fullyConnected



def fully_connected_fine_tuning(InputShape):
    ''' '''
    fullyConnected = Sequential()
    fullyConnected.add(GlobalAveragePooling2D(input_shape = InputShape))
    fullyConnected.add(Dense(12, activation='softmax'))    
    return fullyConnected



def create_top_model(InputShape, numDenseNeurons = 256, dropout_rate = 0.5):
    ''' 
    Borrowed from:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    '''
    top_model = Sequential()
    top_model.add(Flatten(input_shape = InputShape))
    top_model.add(Dense(numDenseNeurons, activation='relu'))
    top_model.add(Dropout(dropout_rate))
    top_model.add(Dense(12, activation='sigmoid'))
    return top_model