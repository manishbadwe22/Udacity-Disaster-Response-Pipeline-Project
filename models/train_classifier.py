import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle
import string
from sklearn.metrics import accuracy_score


def load_data(database_filepath):
    """Loads Pandas dataframe from database and converts it to X,Y and list of category names
       Then, In series of subnested functions, cleans the messages. 
    Input:
        database_filepath (str): Filepath of the sqlite database where processed messages and categories are stored
    Output:
        X (pandas dataframe): The messages
        Y (pandas dataframe): Classification categories
        category_names (list): List of the category names
    """
    table_name = 'disaster_response_table'
    # load data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
        
    X = df['message']
    Y = df.iloc[:,4:] # trims last four unwanted columns
    
    global category_names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize (text):
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        
    }
    grid_search = GridSearchCV(pipeline, parameters)
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    INPUTs:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    OUTPUT:
        prints out the precision, recall and f1-score
    """
    # Prediction
    Y_pred = model.predict(X_test)
    
    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('---------------------------------')

    for i in range(Y_test.shape[1]):
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """saves the model to the given filepath
    INPUTs:
        model (scikit-learn model): The model
        model_filepath (string): the filepath to save the model
    OUPTUT:
        Saves the model as a pickle file
    """
    #joblib.dump(model, model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))

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