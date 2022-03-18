import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    Reads two csv files, containing data about disaster related
    messages and the categories of the messages, and combines
    them into a single dataframe.

    Parameters
    ----------
    messages_filepath : string -> The path to the csv file
        containing data about disaster related messages.
    categories_filepath : string -> The path to the csv file
        containing data about the categories of the messages.

    Returns
    -------
    df -> The DataFrame combining the two datasets.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    return df


def clean_data(df):

    """
    Cleans the data in the disaster response
    DataFrame by:
    - extracting the 'categories' column from the DataFrame and
      expanding it into a separate 'categories' DataFrame;
    - getting the list of categories and setting them as column
      names in the 'categories' DataFrame;
    - setting each value in the 'categories' DataFrame columns
      to either '1' (if the message belongs to this category) or
      '0' (if the message does not belong to this category);
    - converting the values in the 'categories' DataFrame from
      string to int and replacing erroneous values of '2' to '0';
    - dropping the 'categories' column from the original DataFrame;
    - merging the original DataFrame with the new 'categories'
      DataFrame;
    - dropping duplicates in the merged DataFrame.

    Parameters
    ----------
    df: DataFrame -> The Dataframe containing data about the disaster
        related messages and their categories.

    Returns
    -------
    df_clean -> The cleaned and processed DataFrame.
    """

    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.str.rstrip('-10')
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    categories = categories.replace({2: 0})
    df.drop(columns='categories', inplace=True)

    df_clean = pd.concat([df, categories], axis=1)
    df_clean.drop_duplicates(inplace=True)

    return df_clean


def save_data(df, database_filename):

    """
    Saves the cleaned DataFrame as a SQLite database.

    Parameters
    ----------
    df: DataFrame -> The cleaned DataFrame.
    database_filename: string -> The name of the SQLite database.

    Returns
    -------
    None.
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():

    """
    Loads, cleans, and saves the data related to disaster
    response messages in a format ready for applying machine
    learning tasks.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """

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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()