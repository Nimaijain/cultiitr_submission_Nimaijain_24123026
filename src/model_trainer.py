import pickle
from surprise import Dataset, Reader, SVD, KNNWithMeans

def train_framework_models(train_df):
    """Fits dual-paradigm models to compute vector profiles."""
    reader = Reader(rating_scale=(1.0, 5.0))
    surprise_data = Dataset.load_from_df(train_df[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset = surprise_data.build_full_trainset()

    print("Fitting SVD Engine...")
    svd = SVD(n_factors=30, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)

    print("Fitting Item-Collaborative Engine...")
    item_cf = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
    item_cf.fit(trainset)

    return svd, item_cf
