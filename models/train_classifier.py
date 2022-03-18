import sys
import pandas as pd
import joblib
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filename):

    """
    Loads the 'messages' table from the SQLite database
    containing the data about disaster related messages,
    saves it into a DataFrame, and splits the DataFrame
    in X and y variables.

    Parameters
    ----------
    database_filename: string -> The name of the SQLite
        database.

    Returns
    -------
    X -> Series containing all disaster related messages.
    y -> DataFrame containing the data about categories
        of the messages.
    category_names - > List of possible categories of
        the disaster related messages.

    """

    engine = create_engine('sqlite:///' + database_filename)
    df = pd.read_sql_table('messages', engine)

    X = df.message
    y = df.iloc[:, 4:]

    category_names = y.columns

    return X, y, category_names


def tokenize(text):

    """
    Applies transformation to the input text to prepare it
    for applying of machine learning algorithms by:
    - lemmatizing each word in the text and appending them
      to a list;
    - lowering all characters in words;
    - removing spaces and newline characters from the words;
    - deleting all 'stop words' from the resulting list.

    Parameters
    ----------
    text: string -> The text to be processed.

    Returns
    -------
    filtered_tokens -> List of processed tokens.
    """

    tokens = word_tokenize(text)

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    filtered_tokens = [clean_token for clean_token in clean_tokens if clean_token not in stop_words]

    return filtered_tokens


def build_model():

    """
    Builds a pipeline that applies the following steps
    to the dataset:
    - customized CountVectorizer transformer;
    - TfidfTransformer;
    - tuned RandomForestClassifier classifier.

    Parameters
    ----------
    None.

    Returns
    -------
    model -> Pipeline object.
    """

    rf = RandomForestClassifier(random_state=17, n_estimators=50, max_depth=10, max_samples=0.8, max_features=0.8)

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf))
    ])

    return model


def evaluate_model(model, X_test, y_test, category_names):

    """
    Uses a pipeline to predict labels in the test messages
    dataset and prints classification report for each category.

    Parameters
    ----------
    model: Pipeline -> The pipeline object to be used for
        predictions;
    X_test: Series -> Series containing disaster related messages;
    y_test: DataFrame -> DataFrame containing the data about
        messages categories;
    category_names: List - > List of possible categories of
        the disaster related messages

    Returns
    -------
    None.
    """

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    for category in category_names:
        report = classification_report(y_test[category], y_pred_df[category], zero_division=0)
        print(category, report)


def save_model(model, model_filepath):

    """
    Saves a pipeline as a pickle object.

    Parameters
    ----------
    model: Pipeline -> The pipeline object to be used for
        predictions;
    model_filepath: string -> The name of and the path to
        the pickle file to be saved.

    Returns
    -------
    None.
    """

    joblib.dump(model, model_filepath)


def main():

    """
    Applies the previously defined functions to:
    - load X, y, and category_names variables from the dataset;
    - split X and y into train and test sets;
    - build the pipeline;
    - train the pipeline on X_train and y_train;
    - prints the evaluation results of the predictions on X_test
      using the pipeline;
    - saves the pipeline as a pickle file.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()