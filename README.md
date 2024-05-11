# Custom KNN Classifier Implementation
This project implements a custom K Nearest Neighbors (KNN) classifier in Python from scratch and compares its performance
with the built-in KNN classifier provided by scikit-learn.
## Overview
K Nearest Neighbors (KNN) is a popular machine learning algorithm used for classification and regression tasks. 
This project focuses on implementing the classification version of KNN. The steps involved in KNN classification are:
1. Calculate the distance between the query instance and all the training samples.
2. Sort the distance and determine the nearest neighbors based on the K parameter.
3. Gather the class labels of the nearest neighbors.
4. Use simple majority of the K nearest neighbors as the prediction value.
## Files
- `knn_from_scratch.py`: Contains the implementation of the custom KNN classifier.
- `main.py`: Implements the comparison between the custom KNN classifier and the scikit-learn's KNN classifier using a sample dataset (`Social_Network_Ads.csv`).
- `Social_Network_Ads.csv`: Sample dataset used for demonstration.
## Dependencies

- numpy
- pandas
- scikit-learn
## Usage

1. Ensure you have the required dependencies installed.
2. Run `main.py` to compare the accuracy of the custom KNN classifier with scikit-learn's KNN classifier.
# Results
The accuracy score of both the custom KNN classifier and scikit-learn's KNN classifier are printed in the console are the both values are same.
