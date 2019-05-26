import pandas as pd
from surprise import Dataset
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import Reader
from surprise import SVD, SlopeOne, BaselineOnly
from surprise import dump, accuracy
import sys
from sklearn.externals import joblib

df = pd.read_csv("jester_dataset_2/jester_ratings.csv")
reader = Reader(rating_scale=(-10,10))
data = Dataset.load_from_df(df[['user_id', 'joke_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

def SVD_alg():
    print('Testing SVD parameters')
    param_grid = {'n_epochs': [12,13], 'lr_all': [0.0013, 0.0015], 'reg_all': [0.05,0.06]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=10, n_jobs=-2, refit=True)

    #runs fit method for all parameter combinations over splits given by cv
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    #export model
    joblib.dump(gs.best_params['rmse'], 'SVD.pkl', compress=1)
    dump.dump('SVD',algo=gs)

def BaselineOnly_als():
    print('Testing BaselineOnly als parameters')
    param_grid = {'bsl_options': {'method': ['als'], 'reg_i': [7, 6.9,7.1], 'reg_u': [7, 6.9,7.1]}}
    gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse'], cv=10, n_jobs=-2, refit=True)
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    #export model
    joblib.dump(gs.best_params['rmse']['bsl_options'], 'BaselineOnly.pkl', compress=1)
    dump.dump('BaselineOnly',algo=gs)

def BaselineOnly_sgd():
    print('Testing BaselineOnly sgd parameters')
    param_grid = {'bsl_options': {'method': ['sgd'], 'learning_rate': [0.00643,0.00646,0.00649], 'n_epochs': [43, 44,45,46,47]}}
    gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse'], cv=10, n_jobs=-2, refit=True)
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    #export model
    joblib.dump(gs.best_params['rmse']['bsl_options'], 'BaselineOnly.pkl', compress=1)
    dump.dump('BaselineOnly',algo=gs)


# train our best algorithm with train/test
print (sys.argv[1])
if(sys.argv[1] == "SVD"):
    SVD_alg()
elif(sys.argv[1] == 'BaselineOnly' and sys.argv[2] == "als"):
    BaselineOnly_als()
elif(sys.argv[1] == 'BaselineOnly' and sys.argv[2] == "sgd"):
    BaselineOnly_sgd()
