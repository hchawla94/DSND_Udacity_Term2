import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories dataset. Also, combine the two datasets
    Inputs:
        (File Path)messages_filepath: path to csv file containing messages
        (File Path)categories_filepath: path to csv file containing categories
    Outputs:
        (DataFrame)df: combined dataframe
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df

def clean_data(df):
    """
    Loads merged dataset and cleans the same
    Inputs:
        (DataFrame)df: merged dataset
    Outputs:
        (DataFrame)df: cleaned dataset
    """
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # Pull-in the last value of the string as the column value and convert it to numeric
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
    # Drop original categories column 
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Load the dataframe and save it
    Inputs:
        (DataFrame)df: dataframe
        (File Path)database_filename: database saving path
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_Messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.head())

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()