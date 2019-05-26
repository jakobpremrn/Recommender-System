import pandas as pd
from surprise import Dataset
from surprise.model_selection import train_test_split, cross_validate
from surprise import Reader
from surprise import accuracy
from surprise import SVD, SlopeOne, BaselineOnly
from surprise import dump, evaluate
import sys
from sklearn.externals import joblib
from surprise.evaluate import evaluate

def SVD_alg():
    print('Using SVD')
    _, alg = dump.load('SVD')
    predictions = alg.test(testset)
    #pred = alg.predict(5,2)
    #print(pred)
    #print(predictions)
    print(accuracy.rmse(predictions))

    dump.dump('SVD_pred', predictions, alg)

def SlopeOne_alg():
    print('Using SlopeOne')
    alg = SlopeOne()
    print(alg)
    alg.fit(trainset)
    predictions = alg.test(testset)
    print(accuracy.rmse(predictions))

def BaselineOnly_alg():
    print('Using BaselineOnly')
    _, alg = dump.load('BaselineOnly')
    predictions = alg.test(testset)
    print(accuracy.rmse(predictions))

    dump.dump('BSL_pred', predictions, alg)

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
