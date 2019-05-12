import pandas as pd
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import accuracy
from surprise import SVD, SlopeOne, BaselineOnly
from surprise import dump
import sys
from sklearn.externals import joblib

def SVD_alg():
    print('Using SVD')
    _, alg = dump.load('BaselineOnly')
    predictions = alg.test(testset)
    print(accuracy.rmse(predictions))


def SlopeOne_alg():
    print('Using SlopeOne')
    alg = SlopeOne()
    alg.fit(trainset)
    predictions = alg.test(testset)
    print(accuracy.rmse(predictions))


def BaselineOnly_alg():
    print('Using BaselineOnly')
    _, alg = dump.load('BaselineOnly')
    predictions = alg.test(testset)
    print(accuracy.rmse(predictions))


# import data
df = pd.read_csv("jester_dataset_2/jester_ratings.csv")
reader = Reader(rating_scale=(-10,10))
data = Dataset.load_from_df(df[['user_id', 'joke_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

print (sys.argv[1])
if(sys.argv[1] == "SVD"):
    SVD_alg()
elif(sys.argv[1] == 'SlopeOne'):
    SlopeOne_alg()
elif(sys.argv[1] == 'BaselineOnly'):
    BaselineOnly_alg()
