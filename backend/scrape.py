from helper_functions import *
from models.neural_net import *
from models.linear_regressor import *

from bs4 import BeautifulSoup as bs
import requests

import pandas as pd
import numpy as np
from datetime import datetime


def get_cars(num_pages):
    res = []
    for i in range(1, num_pages):
        r = requests.get(f'https://www.finn.no/car/used/search.html?model=1.8078.2000501&page={i}&sort=PUBLISHED_DESC')
        soup = bs(r.content, 'html.parser')
        
        res.append(soup)
    return res

def get_features(num_pages):
    pages = get_cars(num_pages)
    key_info = []
    titles = []
    
    current_year = datetime.now().year

    # getting and cleaning key info and titles
    for i in range(len(pages)):
        res = pages[i].find_all('div', {'class': 'ads__unit__content__keys'})
        title = pages[i].find_all('a', {'class': 'ads__unit__link'})
        if len(res) > 0:
            key_info.append(res)
            titles.append(title)

    key_info = flatten(key_info)
    titles = flatten(titles)
    titles = [titles[i].text.upper() for i in range(len(titles))]
        
    # features from key info
    age = []
    kms = []
    price = []
    
    # additional features from titles
    long_range = []
    performance = []
    ordinary = []
    awd = []
    autopilot = []

    for i in range(len(key_info)):
        curr = key_info[i].get_text('div').split('div')
        curr = [extract_numbers(text) for text in curr if extract_numbers(text)]
        title = titles[i]
        
        if (len(curr) == 3) and (curr[2] > 150000):
            age.append(current_year - curr[0])
            kms.append(curr[1])
            price.append(curr[2])
            
            if ('LONG RANGE' or 'LONGRANGE' or 'LR') in title:
                long_range.append(1)
            else: long_range.append(0)

            if ('AUTOPILOT' or 'AP' or 'AUTO PILOT') in title:
                autopilot.append(1)
            else: autopilot.append(0)

            if ('PERFORMANCE') in title:
                performance.append(1)
            else: performance.append(0)

            if ('4WD' or 'AWD' or 'DUAL MOTOR' or 'DUAL ENGINE') in title:
                awd.append(1)
            else: awd.append(0)
            
            if not (('LONG RANGE' or 'LONGRANGE' or 'LR') or
                    ('PERFORMANCE')) in title:
                ordinary.append(1)
            else: ordinary.append(0)
            
    features = np.array([age, kms, price, long_range, performance, ordinary, awd, autopilot])
    
    if not equal_lengths(features):
        print_lengths(features)
        raise Exception('Lengths are not equal')
    
    return features

def dataset_of(features):
    columns = ['Age', 'KMS', 'Price', 'Long Range', 'Performance', 'Ordinary', 'Autopilot', 'AWD']
    return pd.DataFrame(features.T, columns=columns)



if __name__ == '__main__':
    features = get_features(49)
    df = dataset_of(features)
    df.name = 'MODEL 3 DATASET'

    df_long_range = df[df['Long Range'] == 1]
    df_long_range.name = 'LONG RANGE DATASET'
    df_performance = df[df['Performance'] == 1]
    df_performance.name = 'PERFORMANCE DATASET'

    datasets_list = [df, df_long_range, df_performance]
    linear_regression_on(datasets_list, show_graph=False)
    neural_net_on(datasets_list, show_training=False, show_graph=False)
