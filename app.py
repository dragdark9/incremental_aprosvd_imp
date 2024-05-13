import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recomender_utils import *

# Initialize session state
session_state = st.session_state

# Load datasets
df = pd.read_csv('ratings.csv')
m_info = pd.read_csv("movies.csv")

# Streamlit app layout
def main():
    if 'df_1' not in session_state:
        session_state.df_1 = None
    if 'df_2' not in session_state:
        session_state.df_2 = None
    if 'u1' not in session_state:
        session_state.u1 = None
    if 's1' not in session_state:
        session_state.s1 = None
    if 'vt1' not in session_state:
        session_state.vt1 = None
    if 'new_it_ind' not in session_state:
        session_state.new_it_ind = None
    if 'new_ind_it' not in session_state:
        session_state.new_ind_it = None
    if 'iteration' not in session_state:
        session_state.iteration = 0

    st.title("Recommendation System Streaming Simulation")

    # Sidebar for parameter inputs
    st.sidebar.header("Parameters")
    c1_rate = st.sidebar.slider("c1 Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.01)
    c2_rate = st.sidebar.slider("c2 Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.01)
    k = st.sidebar.number_input("k", value=400)
    initial_data_split = st.sidebar.slider("Initial Data Split", min_value=0.05, max_value=0.8, value=0.3, step=0.05)
    user_id = st.sidebar.number_input("User ID", min_value=1, max_value=6040, value=42)
    n = st.sidebar.number_input("Number of recomended movies", min_value=1, max_value=300, value=20)
    row_percentage_iteration = st.sidebar.slider("Row Percentage Iteration", min_value=0.05, max_value=1.0, value=0.25, step=0.05)

    # Start streaming simulation button
    if st.sidebar.button("Start Offline Precessing"):
        # Initialize streaming parameters
        session_state.df_1, session_state.df_2 = split_last_movies(df, initial_data_split, "userId", "timestamp", "rating")
        mat_1, _, _, item_to_index_1, index_to_item_1 = create_user_item_matrix_2(session_state.df_1, "userId", "movieId", "rating", 6040)
        p1 = create_probabilitys(mat_1)
        c1 = int(len(mat_1[0]) * c1_rate)
        session_state.u1, session_state.s1, session_state.vt1, session_state.new_it_ind, session_state.new_ind_it = initial_svd_offline_index(mat_1, p1, c1, item_to_index_1, index_to_item_1)

        # Display initialization completed message
        st.sidebar.success("Offline Precessing completed. Click 'Start Streaming' to start streaming iterations.")

    # Process streaming button
    if st.sidebar.button("Start Streaming"):
        if session_state.df_1 is not None and session_state.df_2 is not None and session_state.u1 is not None and session_state.s1 is not None and session_state.vt1 is not None and session_state.new_it_ind is not None and session_state.new_ind_it is not None:
            # Perform streaming
            total_rows = len(session_state.df_2)
            chunk_size = int(total_rows * row_percentage_iteration)
            start_index = 0
            end_index = chunk_size
            while start_index < len(session_state.df_2):
                chunk = session_state.df_2.iloc[start_index:end_index]
                all_data_until_now = pd.concat([session_state.df_1, session_state.df_2.iloc[0:end_index]])

                mat_2, _, _, item_to_index_2, index_to_item_2 = create_user_item_matrix_2(chunk, "userId", "movieId", "rating", 6040)
                p2 = create_probabilitys(mat_2)
                c2 = int(len(mat_2[0]) * c2_rate)

                session_state.u1, session_state.s1, session_state.vt1, merged_it_id, merged_id_it = svd_online_incremental_index(session_state.u1, session_state.s1, session_state.vt1, mat_2, c2, p2, k, session_state.new_it_ind, item_to_index_2, session_state.new_ind_it, index_to_item_2)
                session_state.new_it_ind = merged_it_id
                session_state.new_ind_it = merged_id_it

                # Generate and display plots
                st.subheader(f"Chunk {session_state.iteration}")
                pred_user = pred_one_user(session_state.u1, session_state.s1, session_state.vt1, user_id)
                #plot_preferred_genres_one_user_pie(pred_user, m_info, merged_id_it, n)
                #plot_preferred_genres_pie(user_id, all_data_until_now, m_info, n)
                plot_preferred_genres(pred_user, user_id, all_data_until_now , m_info, merged_id_it, n)
                # Update indices and iteration count
                session_state.iteration += 1
                start_index = end_index
                end_index += chunk_size

        session_state.iteration = 0

# Run the Streamlit app
if __name__ == "__main__":
    main()