from datetime import datetime

import pandas as pd

HEADER = ["label", "comment", "author", "subreddit", "date", "parent"]


def load_dataset(path):
    dataframe = pd.read_csv(path, sep="\t", names=HEADER)
    dataframe['date'] = [datetime.strptime(date, '%Y-%m') for date in list(dataframe['date'])]
    dataframe['year'] = [date.year for date in list(dataframe['date'])]
    dataframe['month'] = [date.month for date in list(dataframe['date'])]
    return dataframe
