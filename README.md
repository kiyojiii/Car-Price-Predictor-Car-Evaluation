# Car Price Predictor

Demo Video: https://youtu.be/t-HhG3EmjW4

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/new_demo.png">


# Aim

This project aims to predict the Price of an used Car by taking it's Company name, it's Model name, Year of Purchase, and other parameters.
This project aims to predict the Saferty Evaluation of Car by taking it's Maintenance Cost, Buying Price, No. Of Doors, and other parameters.

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/predict.png">
<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/evaluation.png">

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

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/predicted.png">

1. This project takes the parameters of a car like: Buying Price, Maintenance Cost, No. of Doors, No. of Persons, Lug_Boot, Estimated Safety evaluation.
2. It then predicts the possible safety evaluation of the car. For example, the image below shows the predicted safety evaluation with the estimated car details. 

<img src="https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/evaluated.png">

## How this project does?

1. First of all the data was downloaded from kaggle (https://kaggle.com) <br>
Link for data: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/quikr_car.csv <br>
Link for data: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/car_eval.csv

2. The data was cleaned (it was super unclean ) and analysed.

3. Then a Linear Regression model was built on top of it which had 0.92 R2_score, Then for the Classification Model Forest Algorithm to achieve an accuracy of 99.422%

Link for notebook: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/Quikr%20Analysis.ipynb <br>
Link for notebook: https://github.com/kiyojiii/Car-Price-Predictor-Car-Evaluation/blob/main/Classification%20Analysis.ipynb

4. This project was given the form of an website built on Flask where we used the Linear Regression model and Classification Model with Random Forest Algorithm to perform predictions.

