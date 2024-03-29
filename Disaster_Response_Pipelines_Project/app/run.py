import json
import plotly
import pandas as pd
import re

import nltk
nltk.download(['stopwords'])

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM Disaster_Messages", engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_counts_rel = df[df['related'] != 0].groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    related_counts = (df.groupby('related').count()['message']/len(df))*100
    related_names = ['related-0','related-1','related-2']
    
    Test = pd.DataFrame(df[df.columns[4:]].sum().sort_values(ascending= False))
    Test.columns =["count"]
    category_names = list(df.columns[5:])
    category_perc = ((df.iloc[:,5:] != 0).sum().values/int(Test.loc['related'].values))*100
    
      
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name = 'Total'
                ),
                Bar(
                    x=genre_names,
                    y=genre_counts_rel,
                    name = 'Related'
                )
            ],

            'layout': {
                'title': 'Comparison of Distribution of Overall and Related Message by Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Related Class (%age)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related Category"
                },
           }
        },
         {
            'data': [
                Bar(
                    x=category_names,
                    y=category_perc
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories by %age (within Related Category)',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -30
                }
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