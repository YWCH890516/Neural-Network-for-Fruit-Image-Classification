# Neural-Network-for-Fruit-Image-Classification

## 1. Purpose
Build a neural network to identify three different types of fruit images and classify them into three classes.

## 2. Data
Please refer to Google Drive. The dataset is encapsulated in `Data.zip`, which contains `Data_train` and `Data_test` for training and testing. Three folders named **Carambola**, **Lychee**, and **Pear** under two folders contain three classes of fruit. In the partition for training, there are 490 images per class. You can partition this data to form the validation set yourself and test the performance of your model through the testing data.

## 3. Packages

1. `pandas`
2. `numpy`
3. `matplotlib.pyplot`
4. `sklearn.decomposition` and `sklearn.metrics`
5. `seaborn`
6. `random`
7. ˋPyTorch ˋ
8. ˋtqdm ˋ

## 4. Flowchart

```plaintext
       Start
         |
         v
    Set Packages
         |
         v
   Set Seed & Device
         |
         v
   Configure PCA & DataLoader
         |
         v
   Data Preprocessing
         |
         v
   Define Neural Network
         |
         v
   Train_model & Test_model
         |
         v
   Plot Loss Curve
         |
         v
   Plot Decision Boundary

