import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

jester_data = pd.read_csv("jester_dataset_2/jester_ratings.csv")
#print(jester_data.head())
ratings_mean_count = pd.DataFrame(jester_data.groupby('joke_id')['rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(jester_data.groupby('joke_id')['rating'].count())
print(ratings_mean_count.head())

'''
sns.set_style('dark')
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)
#plt.show()

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating'].hist(bins=50)
#plt.show()

# Test if movies with a higher number of ratings usually have a high average rating
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)
plt.show()
'''

user_jester_rating = jester_data.pivot_table(index='user_id', columns='joke_id', values='rating')
print(user_jester_rating.head())
'''
# how users liked joke 7
joke7_ratings = user_jester_rating['7 ']

# find similar jokes
jokes_like_joke7 = user_jester_rating.corrwith(joke7_ratings)
corr_joke7 = pd.DataFrame(jokes_like_joke7, columns=['Correlation'])
corr_joke7.dropna(inplace=True)
print(corr_joke7.head())
'''
