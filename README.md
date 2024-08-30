## Credit card Fraud Detection Machine Learning Project

Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 

The dataset contains transactions made by credit cards in September 2013 by European cardholders. 

We have to build a classification model to predict whether a transaction is fraudulent or not

Dataset :
Time: The elapsed time between the first transaction in the dataset and each subsequent transaction, measured in seconds

V1 to V28: These are 28 anonymized features. They are numerical variables that capture various properties of the transactions but have been transformed to protect the confidentiality of the data.

Amount: The monetary value of the transaction. This can be useful for identifying patterns associated with fraudulent transactions.

Class: The target variable, indicating whether a transaction is fraudulent (1) or not (0).


Folders: 

Artifacts folder contains: 
train.csv
test.csv
data.csv 
model.pkl 
preproecessing.pkl file

notebook folder contains:
data -> creditcard.csv
EDA.ipynb
MODEL TRAINING.ipynb

src folder contains:
componets->
   data_ingestion.py
   data_transformation.py
   model_trainer.py

pipeline ->
    exception.py
    logger.py
    utils.py
    




