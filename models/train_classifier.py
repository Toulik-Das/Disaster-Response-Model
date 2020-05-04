import sys,re,pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    '''
    load the database from the given filepath and process them as X, y and category_names
    
    Input: Databased filepath
    
    Output:  X: Disaster messages.
             Y: Disaster categories for the messages.
             category_name:  Disaster category names.
    '''
    
    table_name = 'disaster_reports'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)
    
    X = df["message"]
    y = df.drop(["message","id","genre","original"], axis=1)
    
    category_names = y.columns
    return X, y, category_names
    


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    
    """Tokenize text (a disaster message).
    
    Args:
        text: disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
        
    Returns:
        list. It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detecte URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens
    


def build_model():
    '''
    Building a model, create pipeline and  Set grid parameters and 
    perfrom grid search
    
    Output: Returns the model
    '''    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))))
    ])
    
    parameters = {'clf__estimator__learning_rate': [1,2],
                'clf__estimator__n_estimators': [100, 200], 
                'tfidf__use_idf': (True, False)
            }
    
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model and return the classificatio and accurancy score.
    
    Inputs: Model, X_test, y_test, Catgegory_names
    
    Outputs: Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    


def save_model(model, model_filepath):
    """Save model
    
    Args:
        model:  Contains a sklearn estimator.
        model_filepath(String): Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()