import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from surprise import Dataset, Reader, SVD

# --- Load data ---
df_restaurants = pd.read_csv("cleaned_restaurants.csv")
df_restaurants['categories'] = df_restaurants['categories'].fillna('')

df_reviews = pd.read_json("review_top_1000_users_named_cleaned.json", lines=True)

# --- Content-Based Filtering features ---
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_restaurants['categories'])
numerical = df_restaurants[['stars', 'review_count']].fillna(0)
numerical_scaled = MinMaxScaler().fit_transform(numerical)
cbf_features = hstack([tfidf_matrix, numerical_scaled]).tocsr()

# --- Collaborative Filtering model using SVD ---
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_reviews[['user_id', 'business_id', 'stars']], reader)
trainset = data.build_full_trainset()
cf_model = SVD()
cf_model.fit(trainset)

# --- Content-Based Recommendation ---
def get_cb_recommendations(user_id, top_n=5):
    user_reviews = df_reviews[(df_reviews['user_id'] == user_id) & (df_reviews['stars'] >= 4)]
    liked_businesses = user_reviews['business_id']
    if liked_businesses.empty:
        return pd.DataFrame()
    liked_indices = df_restaurants[df_restaurants['business_id'].isin(liked_businesses)].index
    user_vector = cbf_features[liked_indices].mean(axis=0).A
    similarities = cosine_similarity(user_vector, cbf_features).flatten()
    seen_business_ids = set(df_reviews[df_reviews['user_id'] == user_id]['business_id'])
    sorted_indices = similarities.argsort()[::-1]
    recs = []
    for i in sorted_indices:
        business_id = df_restaurants.iloc[i]['business_id']
        if business_id not in seen_business_ids:
            recs.append(i)
        if len(recs) == top_n:
            break
    result = df_restaurants.iloc[recs][['name', 'categories', 'stars']].copy()
    result['name'] = result['name'].str.slice(0, 35)
    result['categories'] = result['categories'].str.slice(0, 60)
    result.rename(columns={'stars': '⭐'}, inplace=True)
    result['⭐'] = result['⭐'].apply(lambda x: f"{x:.1f}")
    return result


# --- Collaborative Filtering Recommendation ---
def get_cf_recommendations(user_id, top_n=5):
    predictions = [cf_model.predict(user_id, iid) for iid in df_restaurants['business_id']]
    scores = pd.DataFrame({
        'business_id': [p.iid for p in predictions],
        'score': [p.est for p in predictions]
    }).sort_values(by='score', ascending=False)
    seen = df_reviews[df_reviews['user_id'] == user_id]['business_id']
    scores = scores[~scores['business_id'].isin(seen)].head(top_n)
    result = pd.merge(scores, df_restaurants, on='business_id')[['name', 'categories', 'stars']].copy()
    result.rename(columns={'stars': '⭐'}, inplace=True)
    return result

# --- Hybrid Recommendation ---
def get_hybrid_recommendations(user_id, top_n=5, alpha=0.5):
    user_reviews = df_reviews[(df_reviews['user_id'] == user_id) & (df_reviews['stars'] >= 4)]
    liked_businesses = user_reviews['business_id']
    if liked_businesses.empty:
        return pd.DataFrame()
    liked_indices = df_restaurants[df_restaurants['business_id'].isin(liked_businesses)].index
    user_vector = cbf_features[liked_indices].mean(axis=0).A
    cb_similarities = cosine_similarity(user_vector, cbf_features).flatten()
    cb_df = df_restaurants[['business_id']].copy()
    cb_df['cb_score'] = cb_similarities
    cf_preds = [cf_model.predict(user_id, iid) for iid in df_restaurants['business_id']]
    cf_df = pd.DataFrame({
        'business_id': [p.iid for p in cf_preds],
        'cf_score': [p.est for p in cf_preds]
    })
    hybrid_df = pd.merge(cb_df, cf_df, on='business_id')
    hybrid_df['hybrid_score'] = alpha * hybrid_df['cf_score'] + (1 - alpha) * hybrid_df['cb_score']
    seen_business_ids = set(df_reviews[df_reviews['user_id'] == user_id]['business_id'])
    hybrid_df = hybrid_df[~hybrid_df['business_id'].isin(seen_business_ids)]
    top = hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)
    result = pd.merge(top[['business_id']], df_restaurants[['business_id','name', 'categories', 'stars']], on='business_id')
    result = result[['name', 'categories', 'stars']].copy()
    result['name'] = result['name'].str.slice(0, 35)
    result['categories'] = result['categories'].str.slice(0, 60)
    result.rename(columns={'stars': '⭐'}, inplace=True)
    result['⭐'] = result['⭐'].apply(lambda x: f"{x:.1f}")
    return result
# --- Category Search Function ---
def search_by_category(category_input):
    category_input = category_input.strip().lower()
    if not category_input:
        return "Please enter a category to search (e.g. pizza, cafe, bakery)."
    filtered = df_restaurants[df_restaurants['categories'].str.lower().str.contains(category_input, na=False)]
    if filtered.empty:
        return f"No restaurants found for category: {category_input}"
    top5 = filtered.sort_values(by='stars', ascending=False).head(5)
    result = top5[['name', 'categories', 'stars']].copy()
    result.rename(columns={'stars': '⭐'}, inplace=True)
    return "### 🔍 Top 5 Restaurants Matching Category\n" + result.to_markdown(index=False)

# --- Main Recommendation Logic ---
def recommend(username_input, category_input):
    try:
        user_row = df_reviews[df_reviews['username'] == username_input]
        if user_row.empty:
            raise ValueError("Username not found.")
        user_id = user_row['user_id'].iloc[0]

        cb_df = get_cb_recommendations(user_id)
        cf_df = get_cf_recommendations(user_id)
        hybrid_df = get_hybrid_recommendations(user_id)
        category_output = search_by_category(category_input)

        cb_output = "### 📘 Content-Based Recommendations\n" + cb_df.to_markdown(index=False) if not cb_df.empty else "No Content-Based Recommendations found."
        cf_output = "### 👥 Collaborative Filtering Recommendations\n" + cf_df.to_markdown(index=False) if not cf_df.empty else "No Collaborative Filtering Recommendations found."
        hybrid_output = "### 🧠 Hybrid Recommendations\n" + hybrid_df.to_markdown(index=False) if not hybrid_df.empty else "No Hybrid Recommendations found."

        return cb_output, cf_output, hybrid_output, category_output

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg, error_msg, ""

# --- Gradio Interface ---
iface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="👤 Enter Username", placeholder="e.g. James Smith"),
        gr.Textbox(label="🍽️ Search by Category", placeholder="e.g. Pizza, Mexican, Cafe")
    ],
    outputs=[
        gr.Markdown(label="📘 Content-Based"),
        gr.Markdown(label="👥 Collaborative Filtering"),
        gr.Markdown(label="🧠 Hybrid"),
        gr.Markdown(label="🔍 Category Results")
    ],
    title="🍴 Restaurant Recommender",
    description="Enter a name to get personalized restaurant recommendations, or search top-rated places by category.",
    theme="default",
    allow_flagging="never"
)

iface.launch()