import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

jester_data = pd.read_csv("jester_dataset_2/jester_ratings.csv")
print(jester_data.head(10))
# average rating per joke
ratings_mean_count = pd.DataFrame(jester_data.groupby('joke_id')['rating'].mean())
# average rating value and number of ratings
ratings_mean_count['rating_counts'] = pd.DataFrame(jester_data.groupby('joke_id')['rating'].count())
print(ratings_mean_count.head(10))
#print(ratings_mean_count.sort_values(by='rating_counts', ascending=False).head())

sns.set_style('dark')
'''
plt.figure(figsize=(8,6))
plt.title('Num. of ratings per specific joke')
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)
#plt.show()

# plot distribution of ratings
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
plt.title('Average rating per joke')
ratings_mean_count['rating'].hist(bins=50)
#plt.show()
'''
# Test if movies with a higher number of ratings usually have a high average rating
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)
plt.show()


user_jester_rating = jester_data.pivot_table(index='user_id', columns='joke_id', values='rating')
print(user_jester_rating.head())
