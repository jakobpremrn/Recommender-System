import pandas as pd
import sys
from surprise import SVD, BaselineOnly
from surprise import dump, Dataset, Reader
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy

predictions_svd, algo_svd = dump.load('SVD_pred')
predictions_bsl, algo_bsl = dump.load('BSL_pred')

print(rmse(predictions_svd))
print(rmse(predictions_bsl))

df_svd = pd.DataFrame(predictions_svd, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_bsl = pd.DataFrame(predictions_bsl, columns=['uid', 'iid', 'rui', 'est', 'details'])

df_svd['err'] = abs(df_svd.est - df_svd.rui)
df_bsl['err'] = abs(df_bsl.est - df_bsl.rui)

print('Description of SVD')
print(df_svd.describe())
print('Description of BSL')
print(df_bsl.describe())
print('Top 5 elements of SVD')
print(df_svd.head())
print('Top 5 elements of BSL')
print(df_bsl.head())

# predictions of bsl where svd has big error
print('BSL prediction where SVD error was more then 3.15 (75% of SVD err)')
bsl_svd_err = df_bsl[df_svd.err > 3.15].sort_values(by='err')
print(bsl_svd_err)
print(bsl_svd_err.shape)
bsl_svd_err = bsl_svd_err[bsl_svd_err.err < 3.15]
print(bsl_svd_err.shape)
print(bsl_svd_err)

print('Size of svd dataframe')
print(df_svd.shape)
# najdi index od 10 najslabsih rezultatov in izpisi vrstico drugega algoritma kjer se ta index nahaja
print('Predictions of BSL on the 10 worst predictions for SVD ')
print(df_bsl.iloc[df_svd.sort_values(by='err')[-10:].index])
print('Predictions of SVD on the 10 worst predictions for BSL')
print(df_svd.iloc[df_bsl.sort_values(by='err')[-10:].index])

# count the number of predictions for each rating value
plt.figure, (ax1, ax2) = plt.subplots(1,2)

df_svd.est.plot(kind='hist', title='SVD', ax=ax1)
df_bsl.est.plot(kind='hist', title='BaselineOnly', ax=ax2)
plt.show()
