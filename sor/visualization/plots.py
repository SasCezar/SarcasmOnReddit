from collections import Counter

import plotly
import plotly.graph_objs as go
from pandas import DataFrame

IMAGES = True

SARCASM_COLOR = 'rgba(0,200,0,1)'
NO_SARCASM_COLOR = 'rgba(192,192,192,1)'

POS_TAGS = ['PUNCT', 'SYM', 'X', 'ADJ', 'VERB', 'CONJ', 'NUM', 'DET', 'ADV',
            'ADP', 'NOUN', 'PROPN', 'PART', 'PRON', 'SPACE', 'INTJ']

STATS_TAGS = ['WORD_COUNT', 'CHAR_COUNT', 'CAPITAL_CASE_PERCENT', 'ENLONGED_WORD_COUNT']

SENTIMENT_TAGS = ["POS", "NEG", "NEU"]


def pos_bar_plot(df: DataFrame, X, name):
    groupdf = df.groupby(['label']).sum()
    Xs = [y for x in X for y in groupdf.columns.values if y.endswith(x)]
    y_scarcasm = [groupdf[x]['1'] for x in Xs]
    y_no_sarcasm = [groupdf[x]['0'] for x in Xs]

    count_sarcasm = sum(y_scarcasm)
    count_no_sarcasm = sum(y_no_sarcasm)

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    sarcasm = go.Bar(
        x=X,
        y=[i / count_no_sarcasm for i in y_no_sarcasm],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm = go.Bar(
        x=X,
        y=[i / count_sarcasm for i in y_scarcasm],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm, no_sarcasm]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def stats_bar_plot(df: DataFrame, X, name):
    groupdf = df.groupby(['label']).mean()
    Xs = [y for x in X for y in groupdf.columns.values if y.endswith(x)]
    y_scarcasm = [groupdf[x]['1'] for x in Xs]
    y_no_sarcasm = [groupdf[x]['0'] for x in Xs]

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    sarcasm = go.Bar(
        x=X,
        y=[i for i in y_no_sarcasm],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm = go.Bar(
        x=X,
        y=[i for i in y_scarcasm],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm, no_sarcasm]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def remap_sentiment(sentiment):
    if sentiment <= -0.05:
        return "NEG"
    if sentiment >= 0.05:
        return "POS"

    return "NEU"


def sentiment_bar_plot(df: DataFrame, X, name):
    compound_sentiment = list(df['COMMENT_VADER_COMPOUND'])
    labels = list(df['label'])
    comments = list(df['comment'])

    text_sentiment = [remap_sentiment(sentiment) for sentiment in compound_sentiment]

    sarcasm = []
    no_sarcasm = []

    i = 10
    for sentiment, label, comment in zip(text_sentiment, labels, comments):
        if label == '1' and sentiment == "NEG" and i:
            print(comment)
            i -= 1
        if label == '0':
            no_sarcasm.append(sentiment)
        else:
            sarcasm.append(sentiment)

    y_sarcasm = Counter(sarcasm)
    y_no_sarcasm = Counter(no_sarcasm)

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    sarcasm_p = go.Bar(
        x=X,
        y=[y_no_sarcasm[x] for x in X],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm_p = go.Bar(
        x=X,
        y=[y_sarcasm[x] for x in X],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm_p, no_sarcasm_p]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800) if IMAGES else None

    sarcasm = []
    no_sarcasm = []

    sarcasm_scores = {}
    no_sarcasm_scores = {}

    for x in SENTIMENT_TAGS:
        sarcasm_scores[x] = []
        no_sarcasm_scores[x] = []

    for sentiment, text, label, comment in zip(compound_sentiment, text_sentiment, labels, comments):
        if label == '0':
            no_sarcasm.append(sentiment)
            no_sarcasm_scores[text].append(abs(sentiment))
        else:
            sarcasm.append(sentiment)
            sarcasm_scores[text].append(abs(sentiment))

    sentiment_intensity_sarcasm = [sum(sarcasm_scores[x]) / (len(sarcasm_scores[x]) + len(no_sarcasm_scores[x]))
                                   for x in SENTIMENT_TAGS]
    no_sentiment_intensity_sarcasm = [sum(no_sarcasm_scores[x]) / (len(sarcasm_scores[x]) + len(no_sarcasm_scores[x]))
                                      for x in SENTIMENT_TAGS]

    sarcasm_p = go.Bar(
        x=SENTIMENT_TAGS,
        y=sentiment_intensity_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(SENTIMENT_TAGS))
    )
    no_sarcasm_p = go.Bar(
        x=SENTIMENT_TAGS,
        y=no_sentiment_intensity_sarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(SENTIMENT_TAGS))
    )

    data = [sarcasm_p, no_sarcasm_p]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-multi-intensity-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-multi-intensity-bar.png'.format(name), height=650,
                                width=1800) if IMAGES else None

    sarcasm_intensity_l = [abs(x) for x in sarcasm if abs(x)]
    no_sarcasm_intensity_l = [abs(x) for x in no_sarcasm if abs(x)]
    sarcasm_intensity = sum(sarcasm_intensity_l) / len(compound_sentiment)
    no_sarcasm_intensity = sum(no_sarcasm_intensity_l) / len(compound_sentiment)

    sarcasm_p = go.Bar(
        x=['Avarage Sentiment'],
        y=[sarcasm_intensity],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR])
    )
    no_sarcasm_p = go.Bar(
        x=['Avarage Sentiment'],
        y=[no_sarcasm_intensity],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR])
    )

    data = [sarcasm_p, no_sarcasm_p]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-intensity-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-intensity-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def subreddit_frequency(df: DataFrame):
    X = [x for x, y in list(Counter(list(df['subreddit'])).most_common(20))]

    subreddits = df['subreddit']
    labels = df['label']

    y = []
    y_no = []
    for subreddit, label in zip(subreddits, labels):
        if subreddit in X:
            if label == '1':
                y.append(subreddit)
            else:
                y_no.append(subreddit)
    y = Counter(y)
    y_no = Counter(y_no)

    total = {}
    for k in X:
        total[k] = y[k] + y_no[k]

    y_scarcasm = [y[x] / total[x] for x in X]
    y_no_sarcasm = [y_no[x] / total[x] for x in X]

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    subreddits_posts = go.Bar(
        x=X,
        y=[total[k] for k in total],
        name="No Sarcasm",
        marker=dict(
            color=['rgb(0,128,128)'] * len(X))
    )

    data = [subreddits_posts]
    layout = go.Layout(
        title='Distribution of sarcasm on subreddits',
        barmode='group',
        # barmode='stack',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='subreddits-count-bar.html')
    plotly.plotly.image.save_as(fig, 'subreddits-count-bar.png', height=650, width=1800) if IMAGES else None

    sarcasm = go.Bar(
        x=X,
        y=y_no_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm = go.Bar(
        x=X,
        y=y_scarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm, no_sarcasm]
    layout = go.Layout(
        title='Distribution of sarcasm on subreddits',
        barmode='group',
        # barmode='stack',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='subreddits-sarcasm-stacked-bar.html')
    plotly.plotly.image.save_as(fig, 'subreddits-sarcasm-stacked-bar.png', height=650, width=1800) if IMAGES else None


def sarcasm_count(df: DataFrame):
    sarcasm = list(df.loc[df['label'] == '1'])
    no_sarcasm = list(df.loc[df['label'] == '0'])

    size = len(df)
    sarcasm_percent = str(len(sarcasm) / size)
    no_sarcasm_percent = str(len(no_sarcasm) / size)
    fig = {
        "data": [
            {
                "values": [sarcasm_percent, no_sarcasm_percent],
                "labels": ['Sarcasm', 'No Sarcasm'],
                "domain": {"x": [0, 1]},
                "name": "Percentage",
                "hoverinfo": "label+percent",
                "hole": .5,
                "type": "pie",
                'marker': {'colors': [SARCASM_COLOR, NO_SARCASM_COLOR]}
            }
        ],
        "layout": {
            "annotations": [
                {
                    "font": {
                        "size": 50
                    },
                    "showarrow": False,
                    "text": "Comments",
                    "x": 0.5,
                    "y": 0.5
                }
            ]

        }
    }

    plotly.offline.plot(fig, filename="word_tweets_donut.html")
    plotly.plotly.image.save_as(fig, 'word_tweets_donut.png', height=1000, width=1000) if IMAGES else None
