# import libraries
import sys

import pandas as pd
import numpy as np
import nltk
from sqlalchemy import create_engine
import re
import pickle

nltk.download(['punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load data from the database
    Input:
        (File Path) database_filepath: File path of sql database
    Output:
        (DataFrame)X: Message (features)
        (DataFrame)Y: Categories (target)
        (List)category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize the input text and return a cleaned token after lemmatization, stop-word removal, punctuation removal
    Inputs:
        (User Input) text: text input
    Outputs:
        (List) cleaned_tokens: list of cleaned tokens
    """
    # Remove punctuation and convert text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower()
    
    # Lemmatize words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
    
    # Remove stop-words
    stop_words = stopwords.words("english")
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return cleaned_tokens


def build_model():
    '''
    Build a ML pipeline using Count Vectorizer, TF-IDF, AdaBoost Classifier and GridSearchCV
    Input: None
    Output:
        Result of GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                  "clf__estimator__learning_rate" : [1.0, 0.5],
                  "clf__estimator__n_estimators" : [20,50]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Output precision, recall, fscore for all the categories for test set
    Inputs:
        model: a trained model
        X_test: features of test set
        Y_test: target values of test set
        category_names: Labels for 36 categories
    Outputs: None
    """
    Y_pred = model.predict(X_test)
    Overall_acc = (Y_pred == Y_test).mean().mean()
    print('Overall accuracy {0:.2f}% \n'.format(Overall_acc*100))
    
    Y_pred_DF = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('****     *****     ***** \n')
        print('Feature Name: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_DF[column]))

def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))

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