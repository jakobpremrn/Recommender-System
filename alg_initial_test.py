import pandas as pd
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Reader
from surprise import accuracy
import sys

def check_for_args():
    args = sys.argv
    for arg in args:
        if(arg == 'SVD'):
            alg_list.append(SVD())
        elif(arg == 'SVDpp'):
            alg_list.append(SVDpp())
        elif(arg == 'SlopeOne'):
            alg_list.append(SlopeOne())
        elif(arg == 'NMF'):
            alg_list.append(NMF())
        elif(arg == 'NormalPredictor'):
            alg_list.append(NormalPredictor())
        elif(arg == 'KNNBaseline'):
            alg_list.append(KNNBaseline())
        elif(arg == 'KNNBasic'):
            alg_list.append(KNNBasic())
        elif(arg == 'KNNWithMeans'):
            alg_list.append(KNNWithMeans())
        elif(arg == 'KNNWithZScore'):
            alg_list.append(KNNWithZScore())
        elif(arg == 'BaselineOnly'):
            alg_list.append(BaselineOnly())
        elif(arg == 'CoClustering'):
            alg_list.append(CoClustering())

    return alg_list



df = pd.read_csv("jester_dataset_2/jester_ratings.csv")

reader = Reader(rating_scale=(-10,10))
data = Dataset.load_from_df(df[['user_id', 'joke_id', 'rating']], reader)

benchmark = []
alg_list = []
check_for_args()

#algs = SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()
for algorithm in alg_list:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))
