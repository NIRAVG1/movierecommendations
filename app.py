import streamlit as st
import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0a1a 50%, #0a0f1a 100%);
    color: #f1f5f9;
}

/* Hero title */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #13131a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.45) !important;
}

/* Cards */
.rec-card {
    background: #1a1a25;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
    transition: all 0.2s;
}
.rec-card:hover {
    border-color: rgba(124,58,237,0.4);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.rank-num {
    min-width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    color: white;
    box-shadow: 0 4px 12px rgba(124,58,237,0.4);
    flex-shrink: 0;
}
.movie-info { flex: 1; }
.movie-name {
    font-weight: 600;
    font-size: 1rem;
    color: #f1f5f9;
    margin-bottom: 6px;
}
.sim-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sim-bar-bg {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.08);
    border-radius: 5px;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 5px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
}
.sim-pct { font-size: 0.75rem; color: #64748b; }

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #a78bfa;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model loading & caching ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Building recommendation engine… (first run only)")
def load_model():
    # Check if pre-built pkl files exist
    if os.path.exists('movie_list.pkl') and os.path.exists('similarity.pkl'):
        movies_df  = pickle.load(open('movie_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies_df, similarity

    # Build from CSV
    credits = pd.read_csv('tmdb_5000_credits.csv')
    credits.dropna(inplace=True)

    def convert3(obj):
        L, counter = [], 0
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i['name']); counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return [i['name']]
        except Exception:
            pass
        return []

    credits['cast'] = credits['cast'].apply(convert3)
    credits['crew'] = credits['crew'].apply(fetch_director)
    credits['cast'] = credits['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    credits['crew'] = credits['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    credits['tags'] = (credits['cast'] + credits['crew']).apply(lambda x: " ".join(x))

    new = credits[['movie_id', 'title', 'tags']].copy()
    new = new[new['tags'].str.strip() != ''].reset_index(drop=True)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vector)

    pickle.dump(new, open('movie_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

    return new, similarity


def recommend(movie_title, movies_df, similarity, top_n=10):
    matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        matches = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
    if matches.empty:
        return None, []
    idx = matches.index[0]
    matched = movies_df.iloc[idx]['title']
    distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    recs = [{'title': movies_df.iloc[i]['title'], 'score': round(float(s), 4)}
            for i, s in distances[1:top_n+1]]
    return matched, recs


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Discover movies similar to your favourites</div>', unsafe_allow_html=True)

# Load model
movies_df, similarity = load_model()
all_titles = sorted(movies_df['title'].tolist())

# Search
col1, col2 = st.columns([4, 1])
with col1:
    selected = st.selectbox(
        "Search for a movie",
        options=[""] + all_titles,
        index=0,
        label_visibility="collapsed",
        placeholder="🔍  Type a movie name…"
    )
with col2:
    search_clicked = st.button("Find Similar 🎯")

# Quick picks
st.markdown('<div class="section-header">✨ Popular picks</div>', unsafe_allow_html=True)
popular = ['Avatar','The Dark Knight','Inception','Interstellar','Titanic',
           'The Avengers','Batman Begins','Iron Man','The Prestige','Jurassic World']
cols = st.columns(5)
for idx, title in enumerate(popular):
    if cols[idx % 5].button(title, key=f"chip_{idx}"):
        selected = title
        search_clicked = True

# Results
if search_clicked and selected:
    matched, recs = recommend(selected, movies_df, similarity)
    if not recs:
        st.error(f"❌ Movie **'{selected}'** not found. Try a different title.")
    else:
        st.markdown(f'<div class="section-header">🍿 Because you liked &nbsp;<span style="color:#f1f5f9">"{matched}"</span></div>',
                    unsafe_allow_html=True)
        max_score = recs[0]['score'] if recs else 1
        for i, rec in enumerate(recs, 1):
            pct = int(rec['score'] / max_score * 100) if max_score > 0 else 0
            score_pct = f"{rec['score']*100:.1f}%"
            st.markdown(f"""
            <div class="rec-card">
                <div class="rank-num">{i}</div>
                <div class="movie-info">
                    <div class="movie-name">{rec['title']}</div>
                    <div class="sim-row">
                        <div class="sim-bar-bg">
                            <div class="sim-bar-fill" style="width:{pct}%"></div>
                        </div>
                        <div class="sim-pct">{score_pct} match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif search_clicked and not selected:
    st.warning("Please select a movie first.")

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">Built on TMDB 5000 · '
    '<a href="https://github.com/NIRAVG1/movierecommendations" style="color:#a78bfa;">Source</a></p>',
    unsafe_allow_html=True
)
