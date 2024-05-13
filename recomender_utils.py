import numpy as np
import pandas as pd
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
import numpy.linalg as ln
import matplotlib.pyplot as plt
import math as mt
import streamlit as st


def initial_svd_offline_index(mat_1, p1, c1, item_index, index_item):
    n_rows = mat_1.shape[0]
    n_cols_1 = mat_1.shape[1]

    mat_c1 = np.zeros((n_rows, c1))
    samples = np.random.choice(range(n_cols_1), c1, replace=False, p=p1)

    # Update dictionaries based on sampled indices
    new_index_item = {i: index_item[samples[t]] for t, i in enumerate(range(c1))}
    new_item_index = {index_item[samples[t]]: i for t, i in enumerate(range(c1))}

    for t in range(c1):
        mat_c1[:, t] = mat_1[:, samples[t]] / np.sqrt(c1 * p1[samples[t]])

    mat_u1, vec_s1, mat_v1t = ln.svd(mat_c1, full_matrices=False)
    mat_s1 = np.diag(vec_s1)

    return (mat_u1, mat_s1, mat_v1t, new_item_index, new_index_item)


def svd_online_incremental_index(u1, s1, vt, mat_2, c2, p2, k, item_index_1, item_index_2, index_item_1, index_item_2):


  n_rows = u1.shape[0]
  n_cols_1 = vt.shape[1]
  n_cols_2 = mat_2.shape[1]

  mat_c2 = np.zeros((n_rows, c2))
  samples = np.random.choice(range(n_cols_2), c2, replace= False, p= p2)

  # Update dictionaries based on sampled indices
  new_index_item_2 = {i: index_item_2[samples[t]] for t, i in enumerate(range(c2))}
  new_item_index_2 = {index_item_2[samples[t]]: i for t, i in enumerate(range(c2))}


  merged_item_index = {}

    # Update item indices from mat_1
  for item, index in item_index_1.items():
    merged_item_index[item] = index

    # Update item indices from mat_2
  for item, index in new_item_index_2.items():
    merged_item_index[item] = index + n_cols_1  # Adjust index for mat_2

    # Create index item dictionary from merged item index
  merged_index_item = {index: item for item, index in merged_item_index.items()}



  for t in range(c2):
    mat_c2[:,t] = mat_2[:, samples[t]]/np.sqrt(c2 * p2[samples[t]])




  n_cols_2 = mat_c2.shape[1]




  mat_f = np.array(np.hstack((s1, np.dot(u1.T, mat_c2))) )

  mat_uf, vec_sf, mat_vft = ln.svd(mat_f, full_matrices=False)

  # keep rank-k approximation
  mat_uf = mat_uf[:, :k]
  vec_sf = vec_sf[:k]
  mat_vft = mat_vft[:k, :]



  V = vt.T
  Z1 = np.zeros((n_cols_1, n_cols_2))
  Z2 = np.zeros((n_cols_2, V.shape[1]))
  I = np.eye(n_cols_2)
  mat_tmp = np.vstack((
     np.hstack((V, Z1)),
      np.hstack((Z2, I))
    ))
  mat_vk = np.dot(mat_tmp, mat_vft.T)


  # compute U_k and S_k
  mat_uk = np.dot(u1, mat_uf)
  mat_sk = np.diag(vec_sf)


  return (mat_uk , mat_sk , mat_vk.T, merged_item_index, merged_index_item)


def create_probabilitys(mat):
  non_null_n = np.count_nonzero(mat)

  n =  mat.shape[1]
  pm = np.zeros(n)
  for i in range(n):
    pm[i] = np.count_nonzero(mat[:, i]) / float(non_null_n)
  return pm



def  pred_2(u,s, vt):
  dim = (len(s), len(s))
  S = np.zeros(dim, dtype=np.float32)
  for i in range(0, len(s)):
    S[i,i] = mt.sqrt(s[i][i])

  right_part = np.dot(S,vt)

  return np.dot(u,right_part)



def  pred_one_user(u,s, vt, user):
  dim = (len(s), len(s))
  S = np.zeros(dim, dtype=np.float32)
  for i in range(0, len(s)):
    S[i,i] = mt.sqrt(s[i][i])

  right_part = np.dot(S,vt)

  return np.dot(u[user],right_part)



def create_user_item_matrix_2(user_item_rating_data, user_column, item_column, rating_column, total_rows):
    filtered_data = user_item_rating_data[[user_column, item_column, rating_column]]

    # Create a user-item matrix with specified number of rows
    user_item_matrix = filtered_data.pivot_table(index=user_column, columns=item_column, values=rating_column, aggfunc='mean')
    user_item_matrix = user_item_matrix.reindex(range(1, total_rows + 1), fill_value=0)
    user_item_matrix = user_item_matrix.fillna(float(0))

    matrix_values = user_item_matrix.values

    # Get the list of users and items
    users = user_item_matrix.index.tolist()
    items = user_item_matrix.columns.tolist()

    # Create dictionaries to map users and items to their indices
    user_to_index = {user: index for index, user in enumerate(users)}
    index_to_user = {index: user for index, user in enumerate(users)}

    item_to_index = {item: index for index, item in enumerate(items)}
    index_to_item = {index: item for index, item in enumerate(items)}

    return matrix_values, user_to_index, index_to_user, item_to_index, index_to_item



def split_last_movies_1(data, remove_percentage, user_id_column, time_column ):
    # Sort the DataFrame by userId and timestamp to ensure data is in chronological order
    data_sorted = data.sort_values(by=[user_id_column, time_column])

    # Group the DataFrame by userId
    grouped = data_sorted.groupby(user_id_column)

    # Define lists to store the splitted data
    data_kept = []
    data_removed = []

    # Iterate over groups
    for _, group in grouped:
        # Calculate the index to split the group
        split_index = int(len(group) * (1 - remove_percentage))

        # Split the group into two parts
        group_kept = group.iloc[:split_index]
        group_removed = group.iloc[split_index:]

        # Append splitted data to lists
        data_kept.append(group_kept)
        data_removed.append(group_removed)

    # Concatenate the splitted data to create two separate datasets
    kept_data = pd.concat(data_kept)
    removed_data = pd.concat(data_removed)

    return kept_data, removed_data



def split_last_movies(data, remove_percentage, user_id_column, time_column, movie_id_column):
    # Sort the DataFrame by userId and timestamp to ensure data is in chronological order
    data_sorted = data.sort_values(by=[user_id_column, time_column])

    # Group the DataFrame by userId
    grouped = data_sorted.groupby(user_id_column)

    # Define lists to store the splitted data
    data_kept = []
    data_removed = []

    # Iterate over groups
    for _, group in grouped:
        # Calculate the index to split the group
        split_index = int(len(group) * (1 - remove_percentage))

        # Split the group into two parts
        group_kept = group.iloc[:split_index]
        group_removed = group.iloc[split_index:]

        # Identify movies that are present in both datasets and remove them from the removed dataset
        #common_movies = group_kept.merge(group_removed, on=movie_id_column)[movie_id_column]
        #group_removed = group_removed[~group_removed[movie_id_column].isin(common_movies)]

        # Append splitted data to lists
        data_kept.append(group_kept)
        data_removed.append(group_removed)

    # Concatenate the splitted data to create two separate datasets
    kept_data = pd.concat(data_kept)
    removed_data = pd.concat(data_removed)

    return kept_data, removed_data



def plot_preferred_genres_one_user_pie(user_movie_list, movies, index_item_dict, n):
    # Step 1: Get the indices of good recommendations
    good_recommendations_indices = np.argsort(user_movie_list)

    # Step 2: Get movie IDs for good recommendations, skipping non-existent indices
    good_recommendations_movie_ids = [index_item_dict.get(index, None) for index in good_recommendations_indices]

    # Remove None values
    good_recommendations_movie_ids = [movie_id for movie_id in good_recommendations_movie_ids if movie_id is not None]

    good_recommendations_movie_ids = np.array(list(set(good_recommendations_movie_ids)))

    good_recommendations_movie_ids = good_recommendations_movie_ids[-n:]

    # Step 3: Get genres for each movie ID
    genres_list = [movies.loc[movies['movieId'] == movie_id, 'genres'].iloc[0].split('|') for movie_id in good_recommendations_movie_ids]

    # Step 4: Count occurrences of each genre
    genre_counts = {}
    for genres in genres_list:
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Step 5: Sort genre counts by count (descending order)
    sorted_genre_counts = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True))

    # Step 6: Plot the pie chart
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(sorted_genre_counts.values(), labels=sorted_genre_counts.keys(), autopct='%1.1f%%', startangle=140)
    ax.set_title('Predicted Genres for the user')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display plot in Streamlit
    st.pyplot(fig)




def plot_preferred_genres_pie(user_id, user_movie_interaction, movies, n):
    # Filter user-movie interaction dataset to consider only movies with ratings 3 or more for the given user
    filtered_data = user_movie_interaction[(user_movie_interaction['userId'] == user_id) & (user_movie_interaction['rating'] >= 2.5)]

    # Get the top n movies with the most ratings
    top_n_movies = filtered_data['movieId'].value_counts().nlargest(n).index

    # Filter data to include only the top n movies
    filtered_data = filtered_data[filtered_data['movieId'].isin(top_n_movies)]

    # Join filtered dataset with movies dataset to get genre information
    filtered_data_with_genres = filtered_data.merge(movies, on='movieId')

    # Count the occurrences of each genre
    genre_counts = filtered_data_with_genres['genres'].str.split('|').explode().value_counts()

    # Plot the most preferred genres as a pie chart
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Most Preferred Genres for User {} (Top {} Movies)'.format(user_id, n))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display plot in Streamlit
    st.pyplot(fig)



def calculate_performance(dataset, user_movie_matrix, mapping_dict, user_id_column, rating_column, movie_id_column):
    # Initialize variables for performance calculation
    total_squared_error = 0
    total_absolute_error = 0
    total_count = 0
    accuracy = 0

    # Iterate through each row of the dataset
    for index, row in dataset.iterrows():
        user_id = row[user_id_column]
        movie_id = int(row[movie_id_column])
        actual_rating = row[rating_column]

        # Check if the user ID and movie ID exist in the mapping dictionary

        matrix_row_index = mapping_dict[movie_id]

            # Get the predicted rating from the user_movie_matrix
        predicted_rating = user_movie_matrix[int(user_id)-1 , matrix_row_index]

            # Pre-process the predicted rating
        predicted_rating = max(1, min(predicted_rating, 5))

        #predicted_rating = np.round(predicted_rating)

        if abs((predicted_rating -   actual_rating)) < 2:
          accuracy += 1



            # Calculate errors
        squared_error = (actual_rating - predicted_rating) ** 2
        absolute_error = abs(actual_rating - predicted_rating)

            # Update totals
        total_squared_error += squared_error
        total_absolute_error += absolute_error
        total_count += 1

    # Calculate performance measures
    if total_count != 0:
      mean_squared_error = total_squared_error / total_count
      mean_absolute_error = total_absolute_error / total_count
      accuracy = accuracy / total_count
    else:
      mean_squared_error = 0
      mean_absolute_error = 0

    return mean_squared_error, mean_absolute_error, accuracy




def plot_preferred_genres_2(user_movie_list, user_id, user_movie_interaction, movies, index_item_dict, n):
    # Plot the predicted genres for the user
    good_recommendations_indices = np.argsort(user_movie_list)
    good_recommendations_movie_ids = [index_item_dict.get(index, None) for index in good_recommendations_indices]
    good_recommendations_movie_ids = [movie_id for movie_id in good_recommendations_movie_ids if movie_id is not None]
    good_recommendations_movie_ids = np.array(list(set(good_recommendations_movie_ids)))
    good_recommendations_movie_ids = good_recommendations_movie_ids[-n:]

    genres_list = [movies.loc[movies['movieId'] == movie_id, 'genres'].iloc[0].split('|') for movie_id in good_recommendations_movie_ids]
    genre_counts = {}
    for genres in genres_list:
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    sorted_genre_counts = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].pie(sorted_genre_counts.values(), labels=sorted_genre_counts.keys(), autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Predicted Genres for the user')
    axs[0].axis('equal') 

    # Plot the most preferred genres for the user
    filtered_data = user_movie_interaction[(user_movie_interaction['userId'] == user_id) & (user_movie_interaction['rating'] >= 2.5)]
    top_n_movies = filtered_data['movieId'].value_counts().nlargest(n).index
    filtered_data = filtered_data[filtered_data['movieId'].isin(top_n_movies)]
    filtered_data_with_genres = filtered_data.merge(movies, on='movieId')
    genre_counts = filtered_data_with_genres['genres'].str.split('|').explode().value_counts()

    axs[1].pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
    axs[1].set_title('Most Preferred Genres for User {} (Top {} Movies)'.format(user_id, n))
    axs[1].axis('equal') 

    # Display plot in Streamlit
    st.pyplot(fig)



# Function to plot preferred genres as pie charts side by side with consistent colors
def plot_preferred_genres(user_movie_list, user_id, user_movie_interaction, movies, index_item_dict, n):
    # Plot the predicted genres for the user
    good_recommendations_indices = np.argsort(user_movie_list)
    good_recommendations_movie_ids = [index_item_dict.get(index, None) for index in good_recommendations_indices]
    good_recommendations_movie_ids = [movie_id for movie_id in good_recommendations_movie_ids if movie_id is not None]
    good_recommendations_movie_ids = np.array(list(set(good_recommendations_movie_ids)))
    good_recommendations_movie_ids = good_recommendations_movie_ids[-n:]

    genres_list = [movies.loc[movies['movieId'] == movie_id, 'genres'].iloc[0].split('|') for movie_id in good_recommendations_movie_ids]
    genre_counts = {}
    for genres in genres_list:
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    sorted_genre_counts = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Get a fixed set of colors for consistent plotting
    num_colors = 10
    colors = plt.cm.tab10(np.linspace(0, 1, num_colors))

    # Plot the predicted genres for the user
    axs[0].pie(sorted_genre_counts.values(), labels=sorted_genre_counts.keys(), autopct='%1.1f%%', startangle=140, colors=colors)
    axs[0].set_title('Predicted Genres for the user')
    axs[0].axis('equal')

    # Plot the most preferred genres for the user
    filtered_data = user_movie_interaction[(user_movie_interaction['userId'] == user_id) & (user_movie_interaction['rating'] >= 2.5)]
    top_n_movies = filtered_data['movieId'].value_counts().nlargest(n).index
    filtered_data = filtered_data[filtered_data['movieId'].isin(top_n_movies)]
    filtered_data_with_genres = filtered_data.merge(movies, on='movieId')
    genre_counts = filtered_data_with_genres['genres'].str.split('|').explode().value_counts()

    axs[1].pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    axs[1].set_title('Most Preferred Genres for User {} (Top {} Movies)'.format(user_id, n))
    axs[1].axis('equal')

    # Display plot in Streamlit
    st.pyplot(fig)