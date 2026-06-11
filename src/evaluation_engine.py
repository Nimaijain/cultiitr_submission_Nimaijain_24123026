import numpy as np
from sklearn.metrics import mean_squared_error
from collections import defaultdict

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_map_at_10(test_df, svd_model, relevance_threshold=3.5):
    user_preds = defaultdict(list)
    for _, row in test_df.iterrows():
        uid, mid, actual = str(row['User_ID']), str(row['Movie_ID']), row['Rating']
        predicted = svd_model.predict(uid, mid).est
        user_preds[uid].append((predicted, actual))
        
    avg_precisions = []
    for uid, user_ratings in user_preds.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_10 = user_ratings[:10]
        
        relevant_count = 0
        precision_score = 0.0
        
        for rank, (pred, actual) in enumerate(top_10, start=1):
            if actual >= relevance_threshold:
                relevant_count += 1
                precision_score += (relevant_count / rank)
                
        if relevant_count > 0:
            avg_precisions.append(precision_score / min(len(user_ratings), 10))
        else:
            avg_precisions.append(0.0)
            
    return np.mean(avg_precisions) if avg_precisions else 0.0
