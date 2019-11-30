import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    loads:
    The specified message and category data
    
    Arguments:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories csv
    Returns:
        df (pandas dataframe): Merged messages and categories dataframe
      """  
    
    global categories
    messages = pd.read_csv(messages_filepath)
    # To avoid error of local variable 'categories' referenced before assignment
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    
    """ This functions cleans the data by
        a. converting categories into columns and their values into binary
        b. duplicates removal
    
    Arguments:
        df (pandas dataframe): Merged categories and messages dataframe
    Returns:
        df (pandas dataframe): Cleaned dataframe with category columns with binary values
    """
# Expand the CSV with ; as seperator
    # categories = categories.categories.str.split(';', expand=True)
    categories = df['categories'].str.split(pat=';', expand=True)

# Extract the 36 category names as a list by using lambda function to remove the last 2 letters and apply them as the column names
    row = categories[:1]
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames
 
# Convert category values to just numbers 0 or 1 by using lambda function to keep just the last string and converting it to int
    categories = categories.applymap(lambda x: int(x[-1]))
    
#Transform non binary values to zero for column related which has a value '2'
    categories.related.replace(2, 0, inplace=True)

# Replace categories column in df with new category columns.
    
    # Drop the categories column from the df dataframe since it is no longer needed.
    df.drop('categories', axis=1, inplace=True)
        
    # Concatenate df and categories data frames.
    df = pd.concat([df, categories], axis=1)
    
# Remove duplicates.

    # Check how many duplicates are in this dataset.
    print('Duplicates:', df.duplicated().sum())
    
    # Drop the duplicates.
    df.drop_duplicates(inplace=True)
    
    # Confirm duplicates were removed.
    print('Remaining Duplicates:', df.duplicated().sum())
    
    return df

def save_data(df, database_filename):
    
    '''
    Arguments: 
        df: Clean Dataframe to be saved
        database_filepath - Filepath for saving the database   
    Returns
        Saved database
    '''
   
    table_name = 'disaster_response_table'
    # create engine 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists 
    df.to_sql(table_name, engine, index=False, if_exists='replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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