

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# #

data=pd.read_csv("C:/Users/FATHIMA SHEMEEMA/Music/movie_project/IMDB_Flask/movie_metadata.csv")
#data.head(5)

# Set the option to display all columns
pd.set_option('display.max_columns', None)

df = pd.DataFrame(data)

# df.head(5)

# df.shape

# df.info()

# df.describe()

# df.isna().sum()

# #percentage null values
# df.isnull().sum()/len(df)*100

# df.columns

# df['color'].value_counts()

# df['country'].unique()

# df['country'].nunique()

# df['language'].unique()

# df['language'].nunique()

# df.isna().sum()

# df['director_name'].nunique()

# df['director_name'].unique()

# Split genres and actors into lists
df['genre'] = df['genres'].str.split(',')

# df.head(5)

# Explode the lists into separate rows for genres
df = df.explode('genre')

# df.head(5)

# df.shape

#drop genres column
df.drop(columns=['genres'],inplace=True)

# df.head(5)

# df.shape

# df.duplicated().sum()

#remove duplicate values

df.drop_duplicates(inplace=True)

df.duplicated().sum()

#Total null values present in each column
# df.isnull().sum()

# histplot=df.hist(figsize=(15,15))

# df['title_year'].unique()

# df['title_year'].min(),df['title_year'].max()

# df['title_year'].nunique()

"""Handling null values of numerical columns"""

for i in ['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes',
       'actor_1_facebook_likes', 'gross','num_voted_users', 'cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews',
    'budget', 'title_year', 'actor_2_facebook_likes','aspect_ratio', 'movie_facebook_likes']:
       df[i].fillna(df[i].median(), inplace=True)

df.isna().sum()

# Fill missing categorical columns with mode
df['content_rating'].fillna(df['content_rating'].mode()[0], inplace=True)
df['color'].fillna(df['color'].mode()[0], inplace=True)
df['language'].fillna(df['language'].mode()[0], inplace=True)
df['country'].fillna(df['country'].mode()[0], inplace=True)

df.isna().sum()

#'movie_title','movie_imdb_link' columns are almost unique,so they doesn't contribute in predicting target variable
#plot_keywords

#Dropping 2 columns
df.drop(columns=['movie_title','movie_imdb_link','plot_keywords'],inplace=True)

df.shape

columns_to_consider = ['director_name', 'actor_2_name', 'actor_1_name',"actor_3_name"]

# Drop rows where any of these columns have NaN values in place
df.dropna(subset=columns_to_consider, inplace=True)

df.isna().sum()

df.shape

"""Outliers Handling"""

numerical_columns=['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes',
       'actor_1_facebook_likes', 'gross','num_voted_users', 'cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews',
    'budget', 'title_year', 'actor_2_facebook_likes','aspect_ratio', 'movie_facebook_likes']

# #checking outliers
# for col in numerical_columns:
#   plt.figure(figsize=(5,5))
#   sns.boxplot(df[col])
#   plt.show()

# df['duration'].max(),df['duration'].min()



"""Feature Engineering"""

#Categorising the target varible
bins = [ 1, 5, 7, 10]
labels = ['FLOP', 'AVG', 'HIT']
df['imdb_binned'] = pd.cut(df['imdb_score'], bins=bins, labels=labels)

#barplot of imbd_binned column

# df.groupby(['imdb_binned']).size().plot(kind="bar",fontsize=14)
# plt.xlabel('Categories')
# plt.ylabel('Number of Movies')
# plt.title('Categorization of Movies')

#Checking the new column
# df.head(5)

"""EDA"""

# df.columns

# genre_count_per_year = df.groupby(['title_year', 'genre']).size().unstack().fillna(0)
# plt.figure(figsize=(14, 7))
# sns.lineplot(data=genre_count_per_year)
# plt.title('Number of Movies Released per Year by Genre')
# plt.xlabel('Year')
# plt.ylabel('Number of Movies')
# plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# df['genre'].nunique()

# total_movies_per_genre = df['genre'].value_counts()
# most_successful_genre = total_movies_per_genre.idxmax()
# print(f"The most common genre is: {most_successful_genre}")

# genre_counts_per_year = df.groupby(['title_year', 'genre']).size()
# top_genres_per_year = genre_counts_per_year.groupby(level=0, group_keys=False).nlargest(5)

# # Print top 5 genres per year
# print("Top 5 Genres per Year:")
# for year, group in top_genres_per_year.groupby(level=0):
#     print(f"\nYear: {year}")
#     print(group.droplevel(0).sort_values(ascending=False))

# top_actors = df['actor_1_name'].value_counts().head(15).index
# top_actor_data = df[df['actor_1_name'].isin(top_actors)]

# # Temporal analysis for top actors
# top_actor_count_per_year = df.groupby(['title_year', 'actor_1_name']).size().unstack().fillna(0)

# plt.figure(figsize=(14, 7))
# sns.lineplot(data=top_actor_count_per_year)
# plt.title('Number of Movies Released per Year by Top Actors')
# plt.xlabel('Year')
# plt.ylabel('Number of Movies')
# plt.legend(title='Actors', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# total_movies_per_actor = top_actor_data['actor_1_name'].value_counts()
# most_common_actor = total_movies_per_actor.idxmax()
# print(f"The actor with most movies is: {most_common_actor}")

# actor_counts_per_year = df.groupby(['title_year', 'actor_1_name']).size()
# top_actors_per_year = actor_counts_per_year.groupby(level=0, group_keys=False).nlargest(5)

# print("\nTop 5 Actors per Year:")
# for year, group in top_actors_per_year.groupby(level=0):
#     print(f"\nYear: {year}")
#     print(group.droplevel(0).sort_values(ascending=False))



#Drectors name

# top_director = df['director_name'].value_counts().head(15).index
# top_director_data = df[df['director_name'].isin(top_director)]

# # Temporal analysis for top actors
# top_director_count_per_year = df.groupby(['title_year', 'director_name']).size().unstack().fillna(0)

# plt.figure(figsize=(14, 7))
# sns.lineplot(data=top_director_count_per_year)
# plt.title('Number of Movies Released per Year by Top Directors')
# plt.xlabel('Year')
# plt.ylabel('Number of Movies')
# plt.legend(title='Directors', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# total_movies_per_director = top_director_data['director_name'].value_counts()
# most_common_director = total_movies_per_director.idxmax()
# print(f"The director with most movies is: {most_common_director}")

# director_counts_per_year = df.groupby(['title_year', 'director_name']).size()
# top_directors_per_year = director_counts_per_year.groupby(level=0, group_keys=False).nlargest(5)

# print("\nTop 5 Directors per Year:")
# for year, group in top_directors_per_year.groupby(level=0):
#     print(f"\nYear: {year}")
#     print(group.droplevel(0).sort_values(ascending=False))

# Filter hit movies
# hit_movies = df[df['imdb_score'].astype(int) >= 6]

# hit_actor_data = hit_movies.explode('actor_1_name')

# hit_actor_data.head(5)

# hit_movies = df[df['imdb_score'].astype(int) >= 6]
# hit_genre_data = hit_movies.explode('genre')
# hit_genre_count_per_year = hit_genre_data.groupby(['title_year', 'genre']).size().unstack().fillna(0)
# plt.figure(figsize=(14, 7))
# sns.lineplot(data=hit_genre_count_per_year)
# plt.title('Number of Hit Movies per Year by Genre')
# plt.xlabel('Year')
# plt.ylabel('Number of Hit Movies')
# plt.legend

# total_hits_per_genre = hit_genre_data['genre'].value_counts()
# most_successful_genre = total_hits_per_genre.idxmax()
# print(f"The most successful genre is: {most_successful_genre}")

# hit_actor_counts_per_year = hit_actor_data.groupby(['title_year', 'actor_1_name']).size()
# top_actors_per_year = hit_actor_counts_per_year.groupby(level=0).idxmax()
# print("Most Successful Actors per Year:")
# for year, actor in top_actors_per_year.items():
#     print(f"Year: {year}, Actor: {actor[1]}, Hit Movies: {hit_actor_counts_per_year[actor]}")

# hit_movies = df[df['imdb_score'].astype(int) >= 6]

# Explode the actors for hit movies

# hit_actor_data = hit_movies.explode('actor_1_name')
# hit_actor_counts_per_year = hit_actor_data.groupby(['title_year', 'actor_1_name']).size()

# # Get the most successful actor for each year
# top_actors_per_year = hit_actor_counts_per_year.groupby(level=0).idxmax()

# # Print the most successful actors per year
# print("Most Successful Actors per Year:")
# for year, actor in top_actors_per_year.items():
#     print(f"Year: {year}, Actor: {actor[1]}, Blockbuster Movies: {hit_actor_counts_per_year[actor]}")

"""Encoding"""

#Describing the categorical data
df.describe(include='object')

#check the unique values of the categorical columns
# for col in df.select_dtypes(include=['object']):
#   print(col)
#   print(df[col].nunique())

# df.head(5)

#Label encoding the categorical columns
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
cat_list=['imdb_binned']
df[cat_list]=df[cat_list].apply(lambda x:le.fit_transform(x))

frequency_enc = ['color',"director_name", 'actor_1_name', 'actor_2_name', 'actor_3_name',
        'genre','language', 'country', 'content_rating' ]

# Apply Frequency Encoding
for col in frequency_enc:
    # Compute frequency of each category in the column
    frequency_map = df[col].value_counts(normalize=True)

    # Map these frequencies to the original column
    df[col] = df[col].map(frequency_map)

# Display the DataFrame to see the encoded columns
# df.head()

# df.dtypes

"""correlation"""

# Create a DataFrame with the selected columns
# correlation_columns = ['color', 'director_name', 'num_critic_for_reviews', 'duration',
#                        'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
#                        'actor_1_facebook_likes', 'gross', 'genre', 'actor_1_name',
#                         'num_voted_users', 'cast_total_facebook_likes',
#                        'actor_3_name', 'facenumber_in_poster',
#                         'num_user_for_reviews', 'language', 'country',
#                        'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',
#                        'imdb_score', 'aspect_ratio', 'movie_facebook_likes','imdb_binned']

# # Create a new DataFrame containing only the columns of interest
# df_correlation = df[correlation_columns]

# # Calculate the correlation matrix
# correlation_matrix = df_correlation.corr()

# # Display the correlation matrix
# correlation_matrix

# Set up the matplotlib figure
# plt.figure(figsize=(14, 10))

# # Create a heatmap using Seaborn
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
#             cbar_kws={"shrink": .8}, linewidths=0.5)

# # Add title and labels
# plt.title("Correlation Heatmap", fontsize=20)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(rotation=0, fontsize=12)

# # Show the plot
# plt.tight_layout()
# plt.show()

#Removing columns due to multicollinearity


df.drop(columns=['cast_total_facebook_likes'],inplace=True)

#Removing the column "imdb_score" since we have "imdb_binned

#I am gonna train the model with imdb_binned not with imdb_score so dropping the column.

#Removing the column "imdb_score" since we have "imdb_binned"
df.drop(columns=['imdb_score'],inplace=True)

df.shape

"""select feature and target"""

df.columns

#Independent Variables
x = df.iloc[:, 0:23].values
#Dependent/Target Variable
y = df.iloc[:, 23].values

y

"""Spliting"""

#split the data
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state = 42,stratify = y)

# print(x_train.shape)
# print(y_train.shape)

#selct the features using RFE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,      # Number of trees in the forest
    max_depth=None,        # Maximum depth of the tree
    min_samples_split=2,   # Minimum number of samples required to split an internal node
    min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node
    max_features='sqrt',   # Use 'sqrt' instead of 'auto'
    bootstrap=True,        # Whether bootstrap samples are used when building trees
    random_state=42   )     # Seed for reproducibility

rfe = RFE(rf, n_features_to_select=15)
x_train_rfe=rfe.fit_transform(x_train, y_train)
x_test_rfe=rfe.transform(x_test)

# Print selected features
selected_features = rfe.support_
feature_names = df.drop(columns=['imdb_binned']).columns
selected_feature_names = feature_names[selected_features]
print("Selected Features for Prediction:", selected_feature_names)

"""scaling"""

#Scaling the dependent variables
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
x_train_rfe_scaled  = sc.fit_transform(x_train_rfe)
x_test_rfe_scaled  = sc.transform(x_test_rfe)

# Train the model with selected features
rf.fit(x_train_rfe_scaled, y_train)

#prediction

y_pred=rf.predict(x_test_rfe_scaled)

y_pred

from sklearn.metrics import accuracy_score


accuracy=rf.score(x_test_rfe_scaled,y_test)
print(f'Accuracy with reduced features: {accuracy:.2f}')

"""Evalution"""

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=labels)
print("Classification Report:\n", report)

# df.head(3)

# # Function to predict movie success based on a new set of features
# def predict_movie_success(features):
#     # Print each feature name along with its corresponding input value
#     for feature_name, feature_value in zip(selected_feature_names, features):
#         print(f"{feature_name}: {feature_value}")

#     features = np.array(features).reshape(1, -1)  # Reshape the input to match model's expected input shape
#     features_scaled = sc.transform(features)  # Scale the features based on selected features
#     prediction = rf.predict(features_scaled)  # Predict the outcome
#     return prediction[0]

# # Example usage: Replace these with actual feature values corresponding to the selected features
# example_features = [563.0,40000.0,471220,2007.0,5000.0]  # Replace with actual values
# print(f"The predicted success of the movie is: {predict_movie_success(example_features)}")

# # Dictionary to map numerical values back to categories
# category_mapping = {
#     0: "Avg",
#     1: "Flop",
#     2: "Hit",

# }

# import numpy as np

# # Function to predict movie success based on a new set of features
# def predict_movie_success(features):
#     # Print each feature name along with its corresponding input value
#     for feature_name, feature_value in zip(selected_feature_names, features):
#         print(f"{feature_name}: {feature_value}")

#     features = np.array(features).reshape(1, -1)  # Reshape the input to match model's expected input shape
#     features_scaled = sc.transform(features)  # Scale the features based on selected features
#     numerical_prediction = rf.predict(features_scaled)[0]  # Predict the outcome (numerical value)

#     # Decode the numerical prediction into a categorical value
#     categorical_prediction = category_mapping.get(numerical_prediction, "Unknown")

#     return categorical_prediction

# # Example usage: Replace these with actual feature values corresponding to the selected features
# example_features = [563.0, 40000.0, 471220, 2007.0, 5000.0]  # Replace with actual values
# predicted_success = predict_movie_success(example_features)
# print(f"The predicted success of the movie is: {predicted_success}")





import pickle
pickle.dump(rf,open("model_out.pkl","wb"))
