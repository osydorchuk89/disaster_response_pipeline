import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Pie, Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    tokens = word_tokenize(text)

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    filtered_tokens = [clean_token for clean_token in clean_tokens if clean_token not in stop_words]

    return filtered_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    cat_df = df.iloc[:, 4:].sum().sort_values()

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = cat_df.index
    category_values = cat_df.values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count",
                    'automargin': 'True'
                },
                'xaxis': {
                    'title': "Genre",
                    'automargin': 'True'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_values,
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Categories',
                'yaxis': {
                    'title': "Category of Messages",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Number of Messages",
                    'automargin': True
                },
                'autosize': 'False',
                'width': 1200,
                'height': 700
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()