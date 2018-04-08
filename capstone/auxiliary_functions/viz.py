import matplotlib.pyplot as plt
import numpy as np

# This code borrowed and adapted from Deep Learning course Jupyter notebook:
# 
def count_target_categories(targets, targetCategories):
    N = len(targets)
    M = len(targetCategories)
    counts = [0]*M
    categoryIndices = range(0,M)
    categoryList = targetCategories.tolist()
    for index in range(0,N):
        targetAsList = targets[index].tolist()
        targetIndex = categoryList.index(targetAsList)
        counts[targetIndex] = counts[targetIndex] + 1
    return counts, categoryIndices


def create_bar_chart(targets, targetCategories, titleString = 'Class Distribution'):
    '''
    Generate bar chart of the distribution of the target classes
    
    INPUTS:
        targets             Mx1 or 1xM array of values
        targetCategories    Nx1 or 1xN array of unique values containing all values 
                            of input 'targets'
        titleString         optional string serving as the title of visualization
    '''
    counts, categoryIndices = count_target_categories(targets, targetCategories)
    objects = ('1','2','3','4','5','6','7','8','9','10', '11', '12')
    totalCounts = np.sum(counts)
    plt.bar(categoryIndices, counts / totalCounts, align = 'center')
    plt.title(titleString)
    plt.ylabel('Proportion of Images')
    plt.xlabel('Class')
    plt.xticks(categoryIndices, objects)
    return


def plot_model_acc_loss(topModel_history):
    
    # Code borrowed from: 
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    # Visualize accuracy
    plt.plot(topModel_history.history['acc'])
    plt.plot(topModel_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Visualize loss
    plt.plot(topModel_history.history['loss'])
    plt.plot(topModel_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return

