# importing necessary libraries

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Helper functions

# this function would return a dictionary whose key is the movie id and the value is the name of the movie.
def movieId_to_name(mid):
    names = dict()
    
    for n in mid:
        m_name = movies.loc[n]['title']
        names[n]= m_name
    
    return names

def encode_units(x):
    if(x <=0):
        return 0
    elif(x>=1):
        return 1

# importing the csv file as pandas Dataframes

movies = pd.read_csv('MovieLensmovies.csv')
user_data = pd.read_csv('user_ratings.csv')
movies.set_index('movieId',inplace=True)

# creating a smaller testing sample of the user dataset, 
# the number of users we are observing is 400.

test = user_data[user_data.userId <=400]
test.set_index('userId',inplace=True)


def main():
    
    # first we want to translate the movieId into the name of the movies.
    # t is a dictionary whose keys are the movie id and values are the names of movies.
    t = MovieId_To_Title_Series(test)
    # we would now convert this dictionary structure into a panda series.
    s = pd.Series(data=t)
    # we also reset the index to movieIds and by applying test['Title'] = s, we matches the movieId with correct movie names in 
    # sample user dataset. 
    test.reset_index(inplace=True)
    test.set_index('movieId',inplace=True)
    test['Title'] = s
    
    # and we reset the index 
    test.reset_index(inplace=True)
    
    # we then start organizing the user data by dropping the inefficient data na
    test['Title'] = test['Title'].str.strip()
    test.dropna(axis=0,subset=['userId'],inplace=True)
    test['userId'] = test['userId'].astype(int)
    
    # we then organize the dataframe, row would be userId, while each column represents different movie names. 
    basket = (test.groupby(['userId','Title'])['Title'].count().unstack().reset_index().fillna(0).set_index('userId'))
    
    # with pandas function applymap, we apply binary encoding throughout the dataset to be consistent. 
    # and binary mapping also helps in terms of applying association rule functions.
    basket_sets = basket.applymap(encode_units)
    
    # finally we apply apriori algorithm to find frequent itemsets in the dataset.
    frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
    rule= association_rules(frequent_itemsets,metric='lift',min_threshold=1)
    
    # for example, we want to see some patterns that are reliable so I picked rules with confidence that are .9 or higher.
    r = rule[ (rule['lift'] >= 5) &
       (rule['confidence'] >= 0.90) ]
    
    for i in range(r.shape[0]):
        print(r.iloc[i]['antecedents'], r.iloc[i]['consequents'])
        
        