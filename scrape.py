from helper_functions import *
from models.neural_net import *
from models.linear_regressor import *

from bs4 import BeautifulSoup as bs
import requests
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime


def get_cars(num_pages: int) -> list:
    # += 1 since page 1 doesnt exist
    num_pages += 1
    res = []
    for i in range(num_pages):
        link = f'https://www.finn.no/car/used/search.html?model=1.8078.2000501&page={i}&sort=PUBLISHED_DESC'
        r = requests.get(link)
        soup = bs(r.content, 'html.parser')
        
        res.append(soup)
    return res

def get_features(num_pages: int) -> np.ndarray:
    print('fetching teslas...')
    pages = get_cars(num_pages)
    print('done!')
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
    standard = []


    for i in range(len(key_info)):
        curr = key_info[i].get_text('div').split('div')
        curr = [extract_numbers(text) for text in curr if extract_numbers(text)]
        title = titles[i]
        
        unique_car_condition = not((('LONG RANGE' in title) or ('LR' in title)) and ('PERFORMANCE' in title))
        
        if (len(curr) == 3) and (curr[2] > 150000) and unique_car_condition:
            age.append(current_year - curr[0])
            kms.append(curr[1])
            price.append(curr[2])
            
            # add only if title is unique
            if ('LONG RANGE' in title) or ('LR' in title):
                long_range.append(1)
                performance.append(0)
                standard.append(0)

            elif ('PERFORMANCE' in title):
                long_range.append(0)
                performance.append(1)
                standard.append(0)

            else:
                long_range.append(0)
                performance.append(0)
                standard.append(1)

            
            
    features = np.array([age, kms, price, long_range, performance, standard])
    
    if not equal_lengths(features):
        print_lengths(features)
        raise Exception('Lengths are not equal')
    
    print(f'{len(age)} results')
    return features


def dataset_of(features: np.ndarray) -> pd.DataFrame:
    columns = ['Age', 'KMS', 'Price', 'Long Range', 'Performance', 'Standard']
    return pd.DataFrame(features.T, columns=columns)


if __name__ == '__main__':
    features = get_features(50)

    df = dataset_of(features)
    df.name = 'model3'
    
    # filepath = Path(f'dataframes/{df.name}.csv')  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    # df.to_csv(filepath)  

    df_long_range = df[df['Long Range'] == 1]
    df_long_range = df_long_range.drop(['Performance', 'Standard'], axis=1)
    df_long_range.name = 'long-range'

    # filepath = Path(f'dataframes/{df_long_range.name}.csv')  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    # df.to_csv(filepath)  

    df_performance = df[df['Performance'] == 1]
    df_performance = df_performance.drop(['Long Range', 'Standard'], axis=1)
    df_performance.name = 'performance'

    # filepath = Path(f'dataframes/{df_performance.name}.csv')  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    # df.to_csv(filepath)  

    datasets_list = [df, df_long_range, df_performance]

    #linear_regression_on(datasets_list, show_graph=False)
    print('-'*50)
    neural_net_on(datasets_list, show_training=False, show_graph=False, save_model=True)
