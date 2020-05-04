# Disaster-Response-Model
![Disaster-Preparedness-and-management-842x421](https://user-images.githubusercontent.com/39211262/80916693-f89ec200-8d77-11ea-983f-887d0af9bcd7.jpg)


## 1. Overview

In this project, I'll apply data engineering to analyze disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> to build a model for an API that classifies disaster messages.

_data_ directory contains a data set which are real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## 2. Required libraries
- nltk 3.5
- numpy 1.18.3 
- pandas 1.0.3
- scikit-learn 0.22.2
- sqlalchemy 1.3.16

## 3. Project Components
   
   ### 3.1 ETL Pipeline
   File _data/process_data.py_ contains data cleaning pipeline that:

   - Loads the `messages` and `categories` dataset
   - Merges the two datasets
   - Cleans the data
   - Stores it in a **SQLite database**

  ### 3.2. ML Pipeline
  File _models/train_classifier.py_ contains machine learning pipeline that:

  - Loads data from the **SQLite database**
  - Splits the data into training and testing sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs result on the test set
  - Exports the final model as a pickle file
   
   
