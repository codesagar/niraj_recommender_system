from fastapi import FastAPI
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from collections import defaultdict
from itertools import islice
import time
import threading


## CONFIG PARAMETERS ##

restaurants = pd.read_csv("restaurants.csv")
users = pd.read_csv("users.csv")

## MAPPINGS ##
idx_to_email = {row[0]:row[3] for row in users.itertuples()}
email_to_idx = {row[3]:row[0] for row in users.itertuples()}
idx_to_restaurant = {row[0]:row[1] for row in restaurants.itertuples()}
restaurant_to_idx = {row[1]:row[0] for row in restaurants.itertuples()}


def initial_matix(n_users=1000, n_restaurants=100, sparsity=0.9):
    rating_matrix = np.random.choice([1,3,4,5],size=(n_users,n_restaurants))
    zero_indices = np.random.choice(rating_matrix.shape[1]*rating_matrix.shape[0], replace=False, size=int(rating_matrix.shape[1]*rating_matrix.shape[0]*sparsity))
    rating_matrix[np.unravel_index(zero_indices, rating_matrix.shape)] = 0 
    return rating_matrix


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


rating_matrix = pd.DataFrame(initial_matix(n_users=users.shape[0],n_restaurants=restaurants.shape[0]))
rating_matrix['index'] = list(range(rating_matrix.shape[0]))
ratings = rating_matrix.melt(id_vars='index',var_name='restaurant_id',value_name='rating')
ratings.rename(columns = {'index':'user_id'}, inplace=True)
ratings = ratings[ratings['rating']!=0]
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings, reader)

sim_parameters = {'name': 'cosine', 'user_based': True, 'min_support': 5}
algo = KNNWithMeans(sim_options=sim_parameters, k=10)

trainset = data.build_full_trainset()
algo.fit(trainset)
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=10)
print("predictions generated")

## Initializing API

def thread_function():
    while(True):
        global data, trainset, testset, predictions, top_n
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        top_n = get_top_n(predictions, n=10)
        print("Recompute complete")
        time.sleep(3600)

x = threading.Thread(target=thread_function)
x.start()
app = FastAPI()


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/recompute_recommendations")
async def recompute_recommendations():
    global data, trainset, testset, predictions, top_n
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n(predictions, n=10)
    return "Recompute` completed"


@app.get("/get_recommendation/{user_email}")
async def get_recommendations(user_email):
    user_idx = email_to_idx[user_email]
    print(top_n[user_idx])
    print(idx_to_restaurant[5])
    recommended_yelp_ids = [idx_to_restaurant[recommendation[0]] for recommendation in top_n[user_idx]]
    return recommended_yelp_ids


@app.get("/set_rating/{user_email}/{yelp_id}/{interaction_type}")
async def get_recommendations(user_email,yelp_id,interaction_type):
    global ratings
    print(type(ratings))
    user_tuple = (ratings['user_id'] == email_to_idx[user_email]) & (ratings['restaurant_id'] == restaurant_to_idx[yelp_id])
    if sum(user_tuple):
        previous_score = ratings.loc[user_tuple,'rating'].values[0]
        print("Previous rating was ",previous_score)
    else:
        print("No rating existed")
        previous_score = 0

    if interaction_type == "info_check":
        if previous_score == 0 or previous_score ==3:
            new_score = previous_score + 1
        else:
            new_score = previous_score
    elif interaction_type == "check_in":
        if previous_score <= 1:
            new_score = previous_score + 3
        else:
            new_score = 5
    else:
        return "Invalid interaction type"
        
    print(ratings.shape)
    if previous_score == 0:
        ratings = ratings.append({'user_id':email_to_idx[user_email], 'restaurant_id':restaurant_to_idx[yelp_id], 'rating': new_score}, ignore_index=True)
    else:
        ratings.loc[user_tuple,'rating'] = new_score
    print(ratings.shape)
    
    return "Success"
