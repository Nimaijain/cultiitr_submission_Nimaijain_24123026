import os
import sys
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

import src.data_pipeline as dp
import src.model_trainer as mt
import src.evaluation_engine as ee

if __name__ == "__main__":
    target_path = os.path.join('/content/netflix_data', 'combined_data_1.txt')
    if not os.path.exists(target_path):
        target_path = os.path.join('netflix_data', 'combined_data_1.txt')

    if not os.path.exists(target_path):
        sys.exit(1)

    raw_data = dp.parse_netflix_data(target_path, 'clean_ratings.csv')
    filtered_data = dp.create_optimized_subset(raw_data)
    train_set, test_set = dp.temporal_split(filtered_data)
    
    svd, _ = mt.train_framework_models(train_set)
    
    predictions = [
        svd.predict(str(user), str(movie)).est 
        for user, movie in zip(test_set['User_ID'], test_set['Movie_ID'])
    ]
    
    rmse = ee.calculate_rmse(test_set['Rating'], predictions)
    map_10 = ee.calculate_map_at_10(test_set, svd)
    
    print(f"SVD RMSE: {rmse:.4f}")
    print(f"SVD MAP@10: {map_10:.4f}")
