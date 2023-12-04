# Dog and Cat detection

This project contains code for training, testing, and making predictions with a deep learning model to classify images of dogs and cats. The model utilizes transfer learning with the VGG16 architecture and is implemented using TensorFlow and Keras.

## Training and Testing
Data augmentation is applied to the training set to enhance generalization, and early stopping is implemented to prevent overfitting. The trained model is saved as 'dog_cat_classifier3.h5', and its performance is evaluated on the test set, presenting metrics such as loss, accuracy, confusion matrix, and classification report. The model uses images from Kaggle dataset: https://www.kaggle.com/datasets/tongpython/cat-and-dog?rvi=1 

The predictions script loads the pre-trained model and uses it to make predictions on new images. The code preprocesses the image, makes predictions, and prints whether the image contains a dog or a cat based on a binary threshold.
