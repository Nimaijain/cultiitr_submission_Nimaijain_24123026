import os
import pandas as pd
import numpy as np

def parse_netflix_data(raw_txt_path, output_csv_path, max_lines=10000000):
    if os.path.exists(output_csv_path):
        return pd.read_csv(output_csv_path)

    parsed_rows = []
    current_movie_id = None

    with open(raw_txt_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            line = line.strip()
            if line.endswith(':'):
                current_movie_id = line[:-1]
            else:
                comp = line.split(',')
                if len(comp) == 3:
                    parsed_rows.append([comp[0], current_movie_id, float(comp[1]), comp[2]])

    df = pd.DataFrame(parsed_rows, columns=['User_ID', 'Movie_ID', 'Rating', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv(output_csv_path, index=False)
    return df

def create_optimized_subset(df, min_user_activity=100, min_movie_popularity=200):
    u_counts = df['User_ID'].value_counts()
    m_counts = df['Movie_ID'].value_counts()
    dense_df = df[
        df['User_ID'].isin(u_counts[u_counts >= min_user_activity].index) &
        df['Movie_ID'].isin(m_counts[m_counts >= min_movie_popularity].index)
    ].copy()
    return dense_df

def temporal_split(df, test_ratio=0.2):
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]
