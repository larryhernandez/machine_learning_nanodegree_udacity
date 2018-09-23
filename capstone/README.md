# Plant Seedlings Classifier (Kaggle Competition)

## Software and Libraries
This project makes use of a Deep Learning Base Amazon Machine Image [(AMI)](https://aws.amazon.com/marketplace/pp/B077GCZ4GR?qid=1537672182632&sr=0-2&ref_=srh_res_product_title) hosted by Amazon Web Services (AWS). The AMI comes pre-loaded with common tools for deep learning, including Keras, the conda environment, and several popular python libraries. This project was completed using the pre-installed environment called tensorflow_p36, which comes equipped with python 3.6 and tensorflow.

The package requirements for the environment are listed in the [requirements file](https://github.com/larryhernandez/machine_learning_nanodegree_udacity/blob/master/capstone/requirements/requirements.txt). In case of any problems, a comprehensive list of all packages (including versions) that were installed on the AMI are listed in the file called [list_of_all_packages.txt](https://github.com/larryhernandez/machine_learning_nanodegree_udacity/blob/master/capstone/requirements/list_of_all_packages.txt).


## Data Source
The image data for this project can be found [here](https://www.kaggle.com/c/plant-seedlings-classification/data) in a file called train.zip. This file is saved locally into a directory called "plantImages/fullDataSet/" and manually unzipped through the commandline to yield the following 12 folders:

    "plantImages/fullDataSet/train/Black-grass/"

    "plantImages/fullDataSet/train/Charlock/"

    "plantImages/fullDataSet/train/Cleavers/"

    "plantImages/fullDataSet/train/Common Chickweed/"

    "plantImages/fullDataSet/train/Common wheat/"

    "plantImages/fullDataSet/train/Fat Hen/"

    "plantImages/fullDataSet/train/Loose Silky-bent/"

    "plantImages/fullDataSet/train/Maize/"

    "plantImages/fullDataSet/train/Scentless Mayweed/"

    "plantImages/fullDataSet/train/Shepherds Purse/"

    "plantImages/fullDataSet/train/Small-flowered Cranesbill/"

    "plantImages/fullDataSet/train/Sugar beet/"