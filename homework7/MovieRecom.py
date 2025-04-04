import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import csr_matrix
import multiprocessing

#########################################
# 1. 数据加载与预处理
#########################################
def load_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    if 'tags' not in movies.columns:
        movies['tags'] = movies['genres']
    return ratings, movies

def train_test_split_data(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test

def create_user_item_matrix(train, min_user_ratings=1000, min_movie_ratings=1000):
    user_counts = train['userId'].value_counts()
    movie_counts = train['movieId'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    train_filtered = train[train['userId'].isin(valid_users) & train['movieId'].isin(valid_movies)]
    
    rating_matrix = train_filtered.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return rating_matrix

#########################################
# 2. 协同过滤推荐（基于用户相似度）
#########################################
def compute_user_similarity(rating_matrix):
    # Normalize ratings by subtracting the mean rating for each user
    normalized_matrix = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0).fillna(0)
    # Convert to sparse matrix for faster computation on large, sparse data
    normalized_sparse = csr_matrix(normalized_matrix.values)
    sim = cosine_similarity(normalized_sparse)
    return pd.DataFrame(sim, index=rating_matrix.index, columns=rating_matrix.index)

def user_cf_recommend(user_id, rating_matrix, user_sim, topN=10, top_sim_users=5):
    # Select top similar users (excluding the target user)
    similar_users = user_sim.loc[user_id].drop(user_id).nlargest(top_sim_users)
    # Compute weighted ratings using vectorized multiplication
    weighted_ratings = rating_matrix.loc[similar_users.index].multiply(similar_users.values, axis=0)
    cf_scores = weighted_ratings.sum(axis=0)
    # Exclude movies already rated by the user
    cf_scores = cf_scores[rating_matrix.loc[user_id] == 0]
    recommended = cf_scores.nlargest(topN).index.tolist()
    return recommended

#########################################
# 3. 基于内容的推荐（利用电影标签）
#########################################
def compute_content_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'].fillna(''))
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return content_sim

def content_based_recommend(user_id, rating_matrix, movies, content_sim, topN=10, rating_threshold=4.0):
    user_ratings = rating_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings >= rating_threshold].index.tolist()
    if not liked_movies:
        popular_movies = rating_matrix.sum().sort_values(ascending=False).head(topN).index.tolist()
        return popular_movies

    # Map movieId to index in the movies DataFrame
    movie_id_to_index = pd.Series(movies.index.values, index=movies['movieId'])
    liked_indices = movie_id_to_index.loc[liked_movies].dropna().astype(int).tolist()
    if not liked_indices:
        popular_movies = rating_matrix.sum().sort_values(ascending=False).head(topN).index.tolist()
        return popular_movies

    # Compute average content similarity for all movies
    avg_sim = np.mean(content_sim[:, liked_indices], axis=1)
    scores = pd.Series(avg_sim, index=movies['movieId'])
    # Exclude movies already rated by the user
    scores = scores[~scores.index.isin(user_ratings[user_ratings > 0].index)]
    recommended = scores.nlargest(topN).index.tolist()
    return recommended

#########################################
# 4. 混合推荐：结合协同过滤与基于内容
#########################################
def hybrid_recommend(user_id, rating_matrix, user_sim, movies, content_sim, alpha=0.5, topN=10, rating_threshold=4.0, top_sim_users=50):
    # Vectorized collaborative filtering computation
    similar_users = user_sim.loc[user_id].drop(user_id).nlargest(top_sim_users)
    weighted_ratings = rating_matrix.loc[similar_users.index].multiply(similar_users.values, axis=0)
    cf_scores = weighted_ratings.sum(axis=0)
    cf_scores = cf_scores[rating_matrix.loc[user_id] == 0]

    # Vectorized content-based computation
    user_ratings = rating_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings >= rating_threshold].index.tolist()
    if liked_movies:
        movie_id_to_index = pd.Series(movies.index.values, index=movies['movieId'])
        liked_indices = movie_id_to_index.loc[liked_movies].dropna().astype(int).tolist()
        avg_sim = np.mean(content_sim[:, liked_indices], axis=1)
        content_scores = pd.Series(avg_sim, index=movies['movieId'])
        content_scores = content_scores[~content_scores.index.isin(user_ratings[user_ratings > 0].index)]
    else:
        content_scores = pd.Series(dtype=float)

    # Normalize the scores if not empty
    if not cf_scores.empty:
        cf_norm = cf_scores / cf_scores.max()
    else:
        cf_norm = pd.Series(dtype=float)
    if not content_scores.empty:
        content_norm = content_scores / content_scores.max()
    else:
        content_norm = pd.Series(dtype=float)

    # Combine the normalized scores
    all_movies = set(cf_norm.index).union(content_norm.index)
    hybrid_scores = {m: alpha * cf_norm.get(m, 0) + (1 - alpha) * content_norm.get(m, 0) for m in all_movies}
    hybrid_scores = pd.Series(hybrid_scores)
    recommended = hybrid_scores.nlargest(topN).index.tolist()
    return recommended

#########################################
# 5. 评价函数（召回率计算）
#########################################
def compute_recall(recommendations, test_data, user_id, rating_threshold=4.0):
    actual = set(test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= rating_threshold)]['movieId'])
    if not actual:
        return None
    rec_set = set(recommendations)
    recall = len(rec_set.intersection(actual)) / len(actual)
    return recall

def evaluate_user(args):
    user_id, rating_matrix, user_sim, movies, content_sim, test_data, topN, rating_threshold, top_sim_users = args
    if user_id not in rating_matrix.index:
        return None
    rec_cf = user_cf_recommend(user_id, rating_matrix, user_sim, topN=topN, top_sim_users=top_sim_users)
    rec_content = content_based_recommend(user_id, rating_matrix, movies, content_sim, topN=topN, rating_threshold=rating_threshold)
    rec_hybrid = hybrid_recommend(user_id, rating_matrix, user_sim, movies, content_sim, alpha=0.5, topN=topN, rating_threshold=rating_threshold, top_sim_users=top_sim_users)
    recall_cf = compute_recall(rec_cf, test_data, user_id, rating_threshold)
    recall_content = compute_recall(rec_content, test_data, user_id, rating_threshold)
    recall_hybrid = compute_recall(rec_hybrid, test_data, user_id, rating_threshold)
    return (recall_cf, recall_content, recall_hybrid)

def evaluate_methods(rating_matrix, user_sim, movies, content_sim, test_data, user_list, topN=10, rating_threshold=3.0, top_sim_users=5):
    # Prepare arguments for parallel processing: only include users present in the rating matrix
    args_list = [
        (user_id, rating_matrix, user_sim, movies, content_sim, test_data, topN, rating_threshold, top_sim_users)
        for user_id in user_list if user_id in rating_matrix.index
    ]
    
    pool = multiprocessing.Pool()
    results = pool.map(evaluate_user, args_list)
    pool.close()
    pool.join()
    
    recalls_cf = []
    recalls_content = []
    recalls_hybrid = []
    
    for res in results:
        if res is None:
            continue
        recall_cf, recall_content, recall_hybrid = res
        if recall_cf is not None:
            recalls_cf.append(recall_cf)
        if recall_content is not None:
            recalls_content.append(recall_content)
        if recall_hybrid is not None:
            recalls_hybrid.append(recall_hybrid)
    
    avg_recall_cf = np.mean(recalls_cf) if recalls_cf else 0
    avg_recall_content = np.mean(recalls_content) if recalls_content else 0
    avg_recall_hybrid = np.mean(recalls_hybrid) if recalls_hybrid else 0
    return avg_recall_cf, avg_recall_content, avg_recall_hybrid

#########################################
# 主函数入口
#########################################
def main():
    from sklearn.model_selection import KFold  # Import KFold for 5折交叉验证
    data_dir = "/Volumes/Dreamer1.6/homework/大数据/ml-32m" 
    ratings, movies = load_data(data_dir)
    print("数据加载完毕！")
    
    # 待比较的 top_sim_users 参数值
    for top_sim in [5, 10, 50]:
        print("\n使用协同过滤 top_sim_users = {} 的结果:".format(top_sim))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_recalls_cf = []
        fold_recalls_content = []
        fold_recalls_hybrid = []
        
        for fold, (train_index, test_index) in enumerate(kf.split(ratings)):
            train_data = ratings.iloc[train_index]
            test_data = ratings.iloc[test_index]
            
            # Create the user-item rating matrix from the training fold
            rating_matrix = create_user_item_matrix(train_data)
            if rating_matrix.empty:
                print(f"Fold {fold}: rating matrix is empty, skipping.")
                continue
            
            # Compute user similarity and content similarity
            user_sim = compute_user_similarity(rating_matrix)
            movies_filtered = movies[movies['movieId'].isin(rating_matrix.columns)].reset_index(drop=True)
            content_sim = compute_content_similarity(movies_filtered)
            
            # Evaluate recommendation methods using the current top_sim_users value
            user_list = rating_matrix.index.tolist()
            avg_recall_cf, avg_recall_content, avg_recall_hybrid = evaluate_methods(
                rating_matrix, user_sim, movies_filtered, content_sim, test_data, user_list, topN=10, rating_threshold=5.0, top_sim_users=top_sim
            )
            
            fold_recalls_cf.append(avg_recall_cf)
            fold_recalls_content.append(avg_recall_content)
            fold_recalls_hybrid.append(avg_recall_hybrid)
            
            print("Fold {}: 协同过滤: {:.4f}, 基于内容: {:.4f}, 混合推荐: {:.4f}".format(
                fold, avg_recall_cf, avg_recall_content, avg_recall_hybrid))
        
        overall_cf = np.mean(fold_recalls_cf) if fold_recalls_cf else 0
        overall_content = np.mean(fold_recalls_content) if fold_recalls_content else 0
        overall_hybrid = np.mean(fold_recalls_hybrid) if fold_recalls_hybrid else 0
        
        print("5折交叉验证平均召回率 (top_sim_users = {}):".format(top_sim))
        print("  协同过滤: {:.4f}".format(overall_cf))
        print("  基于内容: {:.4f}".format(overall_content))
        print("  混合推荐: {:.4f}".format(overall_hybrid))


if __name__ == '__main__':
    main()