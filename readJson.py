import json
import gzip
import pandas as pd
import pickle

path = 'Documents/reviews_Musical_Instruments_5.json.gz'
df = getDF(path)
df.to_pickle('Documents/reviews_MusicalInstruments')


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i+=1
    return pd.DataFrame.from_dict(df, orient='index')


