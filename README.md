# Disaster-Response-Model
![1_C51FyB82wHdzRTgEIsYKPw](https://user-images.githubusercontent.com/39211262/80963993-b2a43580-8e2d-11ea-8bee-34232ab64672.jpeg)


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
- flask 1.1.2

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
  
  ### 3.3. Flask Web App

  Running `python run.py` **from app directory** will start the web app where users can enter their query,     i.e., a request message   sent during a natural disaster.


## 4. Compilation
   
 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 4. Conclusion

Some information about training data set as seen on the main page of the web app.
  
## 5. Credits and Acknowledgements

Thanks <a href="https://www.udacity.com" target="_blank">Udacity</a> for letting me use their logo as favicon for this web app and for all the supports. Also thanks to <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> for providing the disaster dataset.
