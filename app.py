"""
Spotify Music Clustering – Streamlit app
Backend: StandardScaler -> KMeans (+ PCA for visualisation), same pipeline as the notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA_PATH = BASE / "dataset.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
META = ["track_name", "artists", "track_genre"]
SLIDER_RANGES = {  # (min, max, default)
    "danceability": (0.0, 1.0, 0.6), "energy": (0.0, 1.0, 0.6),
    "loudness": (-30.0, 2.0, -7.0), "speechiness": (0.0, 1.0, 0.06),
    "acousticness": (0.0, 1.0, 0.3), "instrumentalness": (0.0, 1.0, 0.0),
    "liveness": (0.0, 1.0, 0.18), "valence": (0.0, 1.0, 0.5),
    "tempo": (40.0, 220.0, 120.0),
}
ACCENT = "#1DB954"
CLUSTER_COLORS = ["#1DB954", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6",
                  "#14b8a6", "#ef4444", "#84cc16"]

st.set_page_config(page_title="Music Cluster Studio", page_icon="🎧", layout="wide")


# ----------------------------------------------------------------------------
# Theme (light / dark)
# ----------------------------------------------------------------------------
THEMES = {
    "dark": dict(bg="#0e1117", card="#161b22", sidebar="#0b0f14", text="#e6edf3",
                 muted="#8b949e", border="#2a313c", input="#1c232d"),
    "light": dict(bg="#f6f7f9", card="#ffffff", sidebar="#ffffff", text="#1f2328",
                  muted="#656d76", border="#d8dee4", input="#ffffff"),
}


def apply_theme(mode: str):
    t = THEMES[mode]
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: {t['bg']}; color: {t['text']};
        }}
        [data-testid="stSidebar"] {{ background: {t['sidebar']}; border-right: 1px solid {t['border']}; }}
        .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"] {{ color: {t['text']}; }}
        .stApp [data-testid="stCaptionContainer"], .stApp small {{ color: {t['muted']}; }}
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"], .stTextInput input, .stNumberInput input {{
            background: {t['input']} !important; color: {t['text']} !important;
            border-color: {t['border']} !important;
        }}
        div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {{ background: {t['card']} !important; }}
        div[data-baseweb="popover"] li {{ color: {t['text']} !important; }}
        .stButton > button, .stDownloadButton > button {{
            background: {ACCENT}; color: #fff; border: none; border-radius: 999px;
            padding: .5rem 1.4rem; font-weight: 600;
        }}
        .stButton > button:hover {{ background: #17a34a; color: #fff; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {t['border']}; }}
        .stTabs [data-baseweb="tab"] {{ color: {t['muted']}; }}
        .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}
        [data-testid="stMetric"], .card {{
            background: {t['card']}; border: 1px solid {t['border']};
            border-radius: 14px; padding: 14px 18px;
        }}
        .hero {{
            background: linear-gradient(135deg, {ACCENT}22, #3b82f622);
            border: 1px solid {t['border']}; border-radius: 18px;
            padding: 22px 28px; margin-bottom: 18px;
        }}
        .hero h1 {{ margin: 0; font-size: 2rem; }}
        .hero p {{ margin: 4px 0 0; color: {t['muted']} !important; }}
        .pill {{
            display: inline-block; padding: 4px 14px; border-radius: 999px;
            color: #fff; font-weight: 700; font-size: .9rem;
        }}
        [data-testid="stDataFrame"] {{ border: 1px solid {t['border']}; border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, mode: str, height=420):
    t = THEMES[mode]
    fig.update_layout(
        template="plotly_dark" if mode == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"]), height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ----------------------------------------------------------------------------
# Backend: data + model
# ----------------------------------------------------------------------------
def describe_clusters(centers_z: np.ndarray) -> dict:
    """Auto-name each cluster from its two most extreme standardised features."""
    names = {}
    for c, row in enumerate(centers_z):
        top = np.argsort(-np.abs(row))[:2]
        parts = [("High " if row[i] > 0 else "Low ") + FEATURES[i] for i in top]
        names[c] = " · ".join(parts)
    return names


def train_model(df: pd.DataFrame, k: int) -> dict:
    X = df[FEATURES]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    pca = PCA(n_components=2).fit(Xs)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xs), size=min(5000, len(Xs)), replace=False)
    sil = silhouette_score(Xs[idx], km.labels_[idx])
    return dict(
        k=k, scaler=scaler, kmeans=km, pca=pca, silhouette=float(sil),
        names=describe_clusters(km.cluster_centers_),
        minmax=(X.min().to_dict(), X.max().to_dict()),
    )


@st.cache_resource(show_spinner="Loading data & model…")
def load_everything(csv_path: str, mtime: float, k: int):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + META).reset_index(drop=True)

    model_file = MODEL_DIR / f"music_kmeans_k{k}.joblib"
    bundle = None
    if model_file.exists() and model_file.stat().st_mtime >= mtime:
        bundle = joblib.load(model_file)
    if bundle is None:
        bundle = train_model(df, k)
        joblib.dump(bundle, model_file)

    Xs = bundle["scaler"].transform(df[FEATURES])
    df["Cluster"] = bundle["kmeans"].predict(Xs)
    coords = bundle["pca"].transform(Xs)
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
    return df, bundle


def predict_cluster(bundle: dict, values: dict):
    x = pd.DataFrame([values])[FEATURES]
    xs = bundle["scaler"].transform(x)
    dists = bundle["kmeans"].transform(xs)[0]
    cluster = int(np.argmin(dists))
    # softmax over negative distance -> a readable "match" score
    w = np.exp(-(dists - dists.min()))
    return cluster, w / w.sum(), xs


def recommend(df: pd.DataFrame, bundle: dict, row: pd.Series, n: int) -> pd.DataFrame:
    """Same-cluster songs (as in the notebook), ranked by closeness to the chosen song."""
    pool = df[(df["Cluster"] == row["Cluster"]) & (df["track_name"] != row["track_name"])]
    pool = pool.drop_duplicates(["track_name", "artists"])
    xs_pool = bundle["scaler"].transform(pool[FEATURES])
    xs_song = bundle["scaler"].transform(row[FEATURES].to_frame().T.astype(float))
    pool = pool.assign(distance=np.linalg.norm(xs_pool - xs_song, axis=1))
    return pool.nsmallest(n, "distance")


def norm01(bundle, values: dict):
    mn, mx = bundle["minmax"]
    return [float(np.clip((values[f] - mn[f]) / (mx[f] - mn[f] + 1e-9), 0, 1)) for f in FEATURES]


def radar(bundle, series: dict, mode: str, title: str):
    fig = go.Figure()
    for i, (name, vals, color) in enumerate(series):
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=FEATURES + FEATURES[:1], name=name,
            fill="toself", line=dict(color=color), opacity=0.65))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], showticklabels=False)),
                      title=title)
    return style_fig(fig, mode, 430)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎧 Music Cluster Studio")
    dark = st.toggle("🌙 Dark mode", value=True, key="dark_mode")
    mode = "dark" if dark else "light"
    st.divider()
    k = st.slider("Number of clusters (K)", 2, 8, 4, help="The notebook uses K = 4.")
    retrain = st.button("🔁 Retrain model")

apply_theme(mode)

st.markdown(
    '<div class="hero"><h1>🎧 Music Cluster Studio</h1>'
    "<p>K-Means music classification on Spotify audio features – predict a vibe, "
    "explore clusters, and get similar-song recommendations.</p></div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.warning("`dataset.csv` was not found next to `app.py`. Upload the Spotify tracks dataset to begin.")
    up = st.file_uploader("Upload dataset.csv", type="csv")
    if up is not None:
        DATA_PATH.write_bytes(up.getvalue())
        st.rerun()
    st.stop()

if retrain:
    for f in MODEL_DIR.glob(f"music_kmeans_k{k}.joblib"):
        f.unlink()
    load_everything.clear()

missing = set(FEATURES + META) - set(pd.read_csv(DATA_PATH, nrows=1).columns)
if missing:
    st.error(f"Dataset is missing required columns: {sorted(missing)}")
    st.stop()

df, bundle = load_everything(str(DATA_PATH), DATA_PATH.stat().st_mtime, k)
names = bundle["names"]
color_of = lambda c: CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
pill = lambda c: f'<span class="pill" style="background:{color_of(c)}">Cluster {c}</span>'

with st.sidebar:
    st.divider()
    st.caption(f"**{len(df):,}** songs · **{bundle['k']}** clusters")

tab1, tab2, tab3 = st.tabs(["🎚️ Classify a track", "🔎 Find similar songs", "📊 Explore clusters"])

# ----------------------------------------------------------------------------
# Tab 1: classify from sliders
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Audio features")
        values = {}
        for f in FEATURES:
            lo, hi, dv = SLIDER_RANGES[f]
            values[f] = st.slider(f.capitalize(), lo, hi, dv, step=(hi - lo) / 200, key=f"s_{f}")

    with right:
        cluster, probs, xs = predict_cluster(bundle, values)
        st.subheader("Result")
        st.markdown(
            f'<div class="card">{pill(cluster)} &nbsp; <b>{names[cluster]}</b></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        centroid_orig = bundle["scaler"].inverse_transform(bundle["kmeans"].cluster_centers_[[cluster]])[0]
        centroid_vals = dict(zip(FEATURES, centroid_orig))
        st.plotly_chart(
            radar(bundle, [("Your track", norm01(bundle, values), "#f59e0b"),
                           (f"Cluster {cluster} average", norm01(bundle, centroid_vals), color_of(cluster))],
                  mode, "Your track vs. cluster average"),
            width="stretch",
        )
        prob_df = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in range(bundle["k"])], "Match": probs})
        fig = px.bar(prob_df, x="Cluster", y="Match", color="Cluster",
                     color_discrete_sequence=CLUSTER_COLORS, title="Cluster match")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, mode, 300), width="stretch")

    st.subheader(f"Sample songs from Cluster {cluster}")
    sample = df[df["Cluster"] == cluster].drop_duplicates(["track_name", "artists"])
    st.dataframe(sample[META].sample(min(10, len(sample)), random_state=1),
                 width="stretch", hide_index=True)

# ----------------------------------------------------------------------------
# Tab 2: song lookup + recommendations
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Search a song")
    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Song name", placeholder="e.g. Bad Liar")
    n_rec = c2.slider("Recommendations", 5, 50, 15)

    if query.strip():
        matches = df[df["track_name"].str.contains(query.strip(), case=False, regex=False)]
        matches = matches.drop_duplicates(["track_name", "artists"]).head(50)
        if matches.empty:
            st.info("Song not found. Try a different spelling.")
        else:
            labels = [f"{r.track_name} — {r.artists}  ({r.track_genre})" for r in matches.itertuples()]
            choice = st.selectbox("Select the exact track", range(len(matches)), format_func=lambda i: labels[i])
            song = matches.iloc[choice]
            st.markdown(
                f'<div class="card">🎵 <b>{song.track_name}</b> by {song.artists} &nbsp; '
                f'{pill(int(song.Cluster))} &nbsp; <i>{names[int(song.Cluster)]}</i></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            recs = recommend(df, bundle, song, n_rec)
            col_a, col_b = st.columns([1.3, 1], gap="large")
            with col_a:
                st.markdown("**Recommended songs** (same cluster, nearest first)")
                show = recs[META + ["distance"]].rename(columns={"distance": "similarity gap"})
                st.dataframe(show.round({"similarity gap": 2}), width="stretch", hide_index=True)
            with col_b:
                song_vals = {f: float(song[f]) for f in FEATURES}
                st.plotly_chart(
                    radar(bundle, [(song.track_name, norm01(bundle, song_vals), color_of(int(song.Cluster)))],
                          mode, "Audio profile"),
                    width="stretch",
                )
    else:
        st.caption("Type part of a song title to find it and get recommendations.")

# ----------------------------------------------------------------------------
# Tab 3: cluster explorer
# ----------------------------------------------------------------------------
with tab3:
    counts = df["Cluster"].value_counts().sort_index()
    cols = st.columns(len(counts))
    for c, col in zip(counts.index, cols):
        col.metric(f"Cluster {c}", f"{counts[c]:,}")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        fig = px.bar(x=[f"Cluster {c}" for c in counts.index], y=counts.values,
                     color=[f"Cluster {c}" for c in counts.index],
                     color_discrete_sequence=CLUSTER_COLORS, title="Songs per cluster",
                     labels={"x": "", "y": "Songs"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, mode), width="stretch")
    with g2:
        samp = df.sample(min(6000, len(df)), random_state=42)
        ev = bundle["pca"].explained_variance_ratio_
        fig = px.scatter(
            samp, x="PC1", y="PC2", color=samp["Cluster"].astype(str),
            color_discrete_sequence=CLUSTER_COLORS, opacity=0.55,
            hover_data=["track_name", "artists"],
            title=f"PCA projection ({ev.sum():.0%} variance explained)",
            labels={"color": "Cluster"})
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(style_fig(fig, mode), width="stretch")

    st.subheader("Cluster profiles")
    cz = pd.DataFrame(bundle["kmeans"].cluster_centers_, columns=FEATURES)
    cz.index = [f"Cluster {i}" for i in cz.index]
    fig = px.imshow(cz.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                    text_auto=".2f", title="Standardised centroid values (red = above average)")
    st.plotly_chart(style_fig(fig, mode, 430), width="stretch")
    for c in range(bundle["k"]):
        st.markdown(f"{pill(c)} &nbsp; {names[c]}", unsafe_allow_html=True)

    st.subheader("Browse a cluster")
    pick = st.selectbox("Cluster", sorted(df["Cluster"].unique()), format_func=lambda c: f"Cluster {c} – {names[c]}")
    genres = df[df["Cluster"] == pick]["track_genre"].value_counts().head(10)
    fig = px.bar(x=genres.values, y=genres.index, orientation="h", title="Top genres in this cluster",
                 labels={"x": "Songs", "y": ""}, color_discrete_sequence=[color_of(pick)])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_fig(fig, mode, 380), width="stretch")
    st.dataframe(df[df["Cluster"] == pick].drop_duplicates(["track_name", "artists"])[META].head(100),
                 width="stretch", hide_index=True)
