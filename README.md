# Car Price Predictor

Demo Video: https://youtu.be/NvxaxwPf3Uk

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/demo.png">



# Aim

This project aims to predict the Price of an used Car by taking it's Company name, it's Model name, Year of Purchase, and other parameters.
This project aims to predict the Saferty Evaluation of Car by taking it's Maintenance Cost, Buying Price, No. Of Doors, and other parameters.

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/predict.png">
<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/evaluation.png">

## How to use?

1. Clone the repository
2. Install the required packages in "requirements.txt" file.
3. If "requirements.txt" file doesn't work, manually download the packages needed with the correct version.

Some packages are:
 - numpy 
 - pandas 
 - scikit-learn

3. Run the "application.py" file
And you are good to go. 

# Description

## What this project does?

1. This project takes the parameters of an used car like: Company name, Model name, Year of Purchase, Fuel Type and Number of Kilometers it has been driven.
2. It then predicts the possible price of the car. For example, the image below shows the predicted price of our Toyota Hilux Conquest. 

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/predicted.png">

## How this project does?

1. First of all the data was downloaded from kaggle (https://kaggle.com) 
Link for data: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/quikr_car.csv
Link for data: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/car_eval.csv

2. The data was cleaned (it was super unclean :( ) and analysed.

3. Then a Linear Regression model was built on top of it which had 0.92 R2_score, Then for the Classification Model Forest Algorithm to achieve an accuracy of 99.422%

Link for notebook: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/Quikr%20Analysis.ipynb
Link for notebook: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/master/Classification%20Analysis.ipynb

4. This project was given the form of an website built on Flask where we used the Linear Regression model and Classification Model with Random Forest Algorithm to perform predictions.

