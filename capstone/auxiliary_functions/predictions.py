import numpy as np


def predict_with_tensors(thisModel, test_tensors):
    ''' 
    thisModel makes predictions with tensors, and converts the one-hot-encoded predictions to integer labels
    '''
#    return [np.argmax(thisModel.predict(np.expand_dims(feature, axis=0))) for feature in testBottleneckFeatures]
    return [thisModel.predict(np.expand_dims(feature, axis=0)) for tensor in test_tensors]


def create_submission(output_path, test_preds, CLASSES):
    ''' 
    Write the final predictions to file. Code largely borrowed from:
    https://www.kaggle.com/gaborfodor/seedlings-pretrained-keras-models
    '''
    test = []
    test['category_id'] = test_preds
    test['species'] = [CLASSES[c] for c in test_preds]
    test[['file', 'species']].to_csv('submission.csv', index=False)
    return