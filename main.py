import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('./data/steam.csv')
df1 = df.copy()
# drop unnecesary columns
df1.drop(columns=['release_date', 'english', 'platforms', 'required_age', 'steamspy_tags'], inplace=True)

# data manipulation
duplicates = df1[df1.duplicated(subset=['appid'], keep=False)]
# Group to aggregate all text rows into one
df_aggregated = df1.groupby('appid')['categories'].agg(lambda x: ';'.join(x)).reset_index()
df1.drop_duplicates(subset=['appid'], inplace=True)
df_merged = pd.merge(df1, df_aggregated, on='appid', how='left', suffixes=('_original', '_aggregated'))
df_merged['categories_original'] = df_merged['categories_aggregated']
df_merged = df_merged.drop(columns=['categories_aggregated'])
df_merged.rename(columns={'categories_original':'categories'}, inplace=True)
df1 = df_merged.copy()
df_desc = pd.read_csv('./data/steam_description_data.csv')
df_desc.rename(columns={'steam_appid':'appid'},inplace=True)
final_df = df1.merge(df_desc, on = 'appid', how= 'inner')
df3 = final_df.copy()
# string manipulation
df3['developer'] = df3['developer'].astype(str)
df3['publisher'] = df3['publisher'].astype(str)
df3['developer'] = df3['developer'].str.replace(' ', '').str.split(';')
df3['publisher'] = df3['publisher'].str.replace(' ', '').str.split(';')
df3['owners'] = df3['owners'].apply(lambda x: int(x.split('-')[1]) if '-' in x else int(x))
df3['categories'] = df3['categories'].str.split(';')
df3['genres'] = df3['genres'].str.split(';')
df3['categories'] = df3['categories'].apply(lambda x: [i.replace(' ', '') for i in x])
df3['categories'] = df3['categories'].apply(lambda x: [i.replace('-', '') for i in x])
df3['detailed_description'] = df3['detailed_description'].apply(lambda x: x.split())
df3['about_the_game'] = df3['about_the_game'].apply(lambda x: x.split())
df3['short_description'] = df3['short_description'].apply(lambda x: x.split())

# data normalization
# will not take into account the ratings as they should be normalized together
numerical_columns = ['achievements', 'average_playtime', 'median_playtime', 'owners', 'price']
ratings_df = df3[['positive_ratings', 'negative_ratings']]
# Data scaling
min_max_scaler = preprocessing.MinMaxScaler()
for column in numerical_columns:
    df3[column] = min_max_scaler.fit_transform(df3[[column]])

inputs_scaled=min_max_scaler.fit_transform(ratings_df)
inputs_n=pd.DataFrame(inputs_scaled,columns=ratings_df.columns)
df_merged2 = pd.merge(df3, inputs_n, left_index=True, right_index=True)
df_merged2.drop(columns=['positive_ratings_x','negative_ratings_x'], inplace=True)
df_merged2.rename(columns={'positive_ratings_y':'positive_ratings', 'negative_ratings_y': 'negative_ratings'}, inplace=True)
# taking only the short description into account as its the only one without the <> characters
df_merged2['tags'] = df_merged2['developer'] + df_merged2['publisher'] + df_merged2['categories'] + df_merged2['short_description'] + df_merged2['genres']
# convert the values from a list to a string but add a space in between while also making it all lowercase
df_merged2['tags'] = df_merged2['tags'].apply(lambda x: ' '.join(map(str, x)))
df_merged2['tags'] = df_merged2['tags'].apply(lambda x: x.lower())
df4 = df_merged2[['appid', 'name', 'tags', 'achievements', 'average_playtime', 'median_playtime', 'owners', 'price']].copy()

# data vectorization
# concatenate numerical columns
num_cols = ['achievements', 'average_playtime', 'median_playtime', 'owners', 'price']
scaler = preprocessing.MinMaxScaler()
numerical_data = scaler.fit_transform(df4[num_cols])
numerical_data_sparse = csr_matrix(numerical_data)
# concatenate text and remove stop words
stop_words = ENGLISH_STOP_WORDS
text_data = df4['tags'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
vectorizer = TfidfVectorizer(max_features=5000)
text_data = vectorizer.fit_transform(text_data)
# Finally, the dataframe with all the features normalized and represented as numbers

all_features = hstack([numerical_data_sparse, text_data])

model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
model.fit(all_features)

def recommend(game, top_n=5):
    index = df4.index[df4['name'] == game].tolist()[0]
    distances, indices = model.kneighbors(all_features[index], n_neighbors=top_n+1)
    recommended_indices = indices.flatten()[1:]
    return df4.iloc[recommended_indices]['name'].tolist()

game_to_recommend = "Team Fortress 2"
recommended_games = recommend(game_to_recommend)

print(f"Recommended games for {game_to_recommend}:")
for game in recommended_games:
    print(game)