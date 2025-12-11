import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==================== Page Config ====================
st.set_page_config(
    page_title="2D Persona Projection Map",
    page_icon="🌌",
    layout="wide"
)

# ==================== Cluster Archetype Definitions ====================
CLUSTER_ARCHETYPES = {
    0: {
        "name": "Social Extroverts",
        "code": "SE",
        "color": "#3498DB",
        "traits": ["High fun", "Frequent go-outs", "High dating frequency"],
        "description": "Outgoing, social butterflies who enjoy parties and dating"
    },
    1: {
        "name": "Rational Thinkers",
        "code": "RT",
        "color": "#9B59B6",
        "traits": ["High intelligence", "High ambition", "Goal-oriented"],
        "description": "Intellectually driven, ambitious individuals focused on achievements"
    },
    2: {
        "name": "Appearance-oriented",
        "code": "AO",
        "color": "#E74C3C",
        "traits": ["High self-rated attractiveness", "Confident", "Style-focused"],
        "description": "Confident individuals who prioritize appearance and presentation"
    },
    3: {
        "name": "Warm & Sincere",
        "code": "WS",
        "color": "#2ECC71",
        "traits": ["High sincerity", "Emotional warmth", "Relationship-focused"],
        "description": "Genuine, caring people who value deep emotional connections"
    },
    4: {
        "name": "Quiet Observers",
        "code": "QO",
        "color": "#F39C12",
        "traits": ["Low go-outs", "Reserved", "Thoughtful"],
        "description": "Introverted, reflective individuals who prefer quality over quantity"
    }
}

CLUSTER_COLORS = [arch["color"] for arch in CLUSTER_ARCHETYPES.values()]

# Goal mapping
GOAL_MAP = {
    1: "Fun night out",
    2: "Meet new people",
    3: "Get a date",
    4: "Serious relationship",
    5: "To say I did it",
    6: "Other"
}

# Field mapping
FIELD_MAP = {
    1: "Law", 2: "Math", 3: "Social Science", 4: "Medical/Pharma",
    5: "Engineering", 6: "English/Creative Writing", 7: "History/Religion",
    8: "Business/Econ", 9: "Education", 10: "Bio/Chem/Physics",
    11: "Social Work", 12: "Undergrad", 13: "Political Science",
    14: "Film", 15: "Fine Arts", 16: "Language", 17: "Architecture", 18: "Other"
}

# ==================== Academic Style ====================
ACADEMIC_LAYOUT = dict(
    font=dict(family="Times New Roman, serif", size=12),
    plot_bgcolor='#FAFAFA',
    paper_bgcolor='white',
)

# ==================== Data Loading ====================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Speed Dating Data.csv", encoding='latin-1')
    return df

@st.cache_data
def preprocess_for_projection(df):
    """
    Preprocess data for 2D persona projection.
    Uses actual column names from Speed Dating dataset.
    """
    # Remove rows with missing key identifiers
    df_clean = df.dropna(subset=['iid', 'wave']).copy()

    # Aggregate to participant level (take first occurrence)
    participant_df = df_clean.groupby('iid').first().reset_index()

    # Map actual column names to our variables
    # Self-assessed traits (attr3_1 = how you think you measure up)
    column_mapping = {
        'fun3_1': 'fun_self',      # Self-rated fun
        'intel3_1': 'intel_self',  # Self-rated intelligence
        'sinc3_1': 'sinc_self',    # Self-rated sincerity
        'amb3_1': 'amb_self',      # Self-rated ambition
        'attr3_1': 'attr_self',    # Self-rated attractiveness
        'go_out': 'go_out',        # Go out frequency (1=Several times a week to 7=Never)
        'date': 'date_freq',       # Dating frequency
    }

    # Rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in participant_df.columns:
            participant_df[new_col] = pd.to_numeric(participant_df[old_col], errors='coerce')

    # Ensure required columns exist
    required_new_cols = ['fun_self', 'intel_self', 'sinc_self', 'amb_self', 'attr_self', 'go_out', 'date_freq']
    for col in required_new_cols:
        if col not in participant_df.columns:
            participant_df[col] = np.nan

    # Fill missing values with median by gender
    numeric_cols = required_new_cols + ['age']
    for col in numeric_cols:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            for gender in [0, 1]:
                mask = participant_df['gender'] == gender
                median_val = participant_df.loc[mask, col].median()
                if pd.isna(median_val):
                    median_val = participant_df[col].median()
                participant_df.loc[mask, col] = participant_df.loc[mask, col].fillna(median_val)

    # Convert go_out to extraversion-friendly scale (reverse: 1=high freq becomes high score)
    # Original: 1=Several times a week, 7=Never
    # Reversed: 7=Several times a week, 1=Never
    if 'go_out' in participant_df.columns:
        participant_df['go_out_reversed'] = 8 - participant_df['go_out']

    return participant_df

def standardize_column(series):
    """Standardize a series to z-scores"""
    if series.std() == 0 or series.isna().all():
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / series.std()

@st.cache_data
def compute_custom_axes(participant_df):
    """
    Compute custom interpretable axes:
    X-axis (Extraversion): fun + go_out_reversed + date_freq
    Y-axis (Rationality): intel + amb - sinc
    """
    df = participant_df.copy()

    # Standardize individual components
    z_fun = standardize_column(df['fun_self'])
    z_go_out = standardize_column(df['go_out_reversed']) if 'go_out_reversed' in df.columns else 0
    z_date = standardize_column(df['date_freq'])
    z_intel = standardize_column(df['intel_self'])
    z_amb = standardize_column(df['amb_self'])
    z_sinc = standardize_column(df['sinc_self'])

    # Compute composite scores
    df['extraversion_raw'] = z_fun + z_go_out + z_date
    df['rationality_raw'] = z_intel + z_amb - z_sinc

    # Standardize to [-3, 3] range for visualization
    def scale_to_range(series, min_val=-3, max_val=3):
        if series.std() == 0:
            return pd.Series([0] * len(series), index=series.index)
        z = (series - series.mean()) / series.std()
        return np.clip(z, min_val, max_val)

    df['extraversion'] = scale_to_range(df['extraversion_raw'])
    df['rationality'] = scale_to_range(df['rationality_raw'])

    return df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-Means clustering on the 2D projection"""
    X = df[['extraversion', 'rationality']].dropna().values
    valid_indices = df[['extraversion', 'rationality']].dropna().index

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Create full labels array
    full_labels = pd.Series(index=df.index, dtype=float)
    full_labels[valid_indices] = labels

    score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0

    return full_labels.values, score, kmeans

def create_projection_plot(df, selected_gender=None, age_range=None,
                          selected_wave=None, selected_cluster=None,
                          selected_goal=None, highlight_id=None):
    """Create the 2D persona projection plot with all visual encodings"""

    # Apply filters
    filtered_df = df.copy()

    if selected_gender is not None and len(selected_gender) > 0:
        filtered_df = filtered_df[filtered_df['gender'].isin(selected_gender)]

    if age_range:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) &
            (filtered_df['age'] <= age_range[1])
        ]

    if selected_wave is not None and len(selected_wave) > 0:
        filtered_df = filtered_df[filtered_df['wave'].isin(selected_wave)]

    if selected_cluster is not None:
        filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

    if selected_goal is not None and len(selected_goal) > 0:
        filtered_df = filtered_df[filtered_df['goal'].isin(selected_goal)]

    # Remove rows with missing coordinates
    filtered_df = filtered_df.dropna(subset=['extraversion', 'rationality'])

    if len(filtered_df) == 0:
        return go.Figure(), filtered_df

    # Create the plot
    fig = go.Figure()

    # Add background reference zones with labels
    # Top-right: Rational Extroverts (Social Achievers)
    fig.add_shape(type="rect", x0=0, y0=0, x1=3.2, y1=3.2,
                  line=dict(width=0), fillcolor="rgba(155, 89, 182, 0.08)")
    fig.add_annotation(x=2.5, y=2.7, text="Rational<br>Extroverts",
                      showarrow=False, font=dict(size=10, color='#9B59B6'), opacity=0.6)

    # Top-left: Rational Introverts
    fig.add_shape(type="rect", x0=-3.2, y0=0, x1=0, y1=3.2,
                  line=dict(width=0), fillcolor="rgba(52, 152, 219, 0.08)")
    fig.add_annotation(x=-2.5, y=2.7, text="Rational<br>Introverts",
                      showarrow=False, font=dict(size=10, color='#3498DB'), opacity=0.6)

    # Bottom-left: Warm Introverts
    fig.add_shape(type="rect", x0=-3.2, y0=-3.2, x1=0, y1=0,
                  line=dict(width=0), fillcolor="rgba(46, 204, 113, 0.08)")
    fig.add_annotation(x=-2.5, y=-2.7, text="Warm<br>Introverts",
                      showarrow=False, font=dict(size=10, color='#2ECC71'), opacity=0.6)

    # Bottom-right: Warm Extroverts (Fun-seekers)
    fig.add_shape(type="rect", x0=0, y0=-3.2, x1=3.2, y1=0,
                  line=dict(width=0), fillcolor="rgba(243, 156, 18, 0.08)")
    fig.add_annotation(x=2.5, y=-2.7, text="Warm<br>Extroverts",
                      showarrow=False, font=dict(size=10, color='#F39C12'), opacity=0.6)

    # Plot each cluster
    for cluster_id in sorted(filtered_df['cluster'].dropna().unique()):
        cluster_id = int(cluster_id)
        cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
        arch_info = CLUSTER_ARCHETYPES.get(cluster_id, CLUSTER_ARCHETYPES[0])

        # Determine symbol based on gender (Female=circle, Male=triangle)
        symbols = ['circle' if g == 0 else 'triangle-up' for g in cluster_data['gender']]

        # Determine size based on self-rated attractiveness
        sizes = []
        for attr in cluster_data['attr_self']:
            if pd.notna(attr):
                sizes.append(max(6, min(20, attr * 1.8)))
            else:
                sizes.append(10)

        # Create rich hover text
        hover_texts = []
        custom_data = []
        for idx, row in cluster_data.iterrows():
            gender_str = 'Female' if row['gender'] == 0 else 'Male'
            goal_str = GOAL_MAP.get(row.get('goal', 0), 'Unknown')
            field_str = FIELD_MAP.get(row.get('field_cd', 0), 'Unknown')

            hover_text = (
                f"<b>ID: {int(row['iid'])}</b><br>"
                f"<b>Age:</b> {row['age']:.0f}<br>"
                f"<b>Gender:</b> {gender_str}<br>"
                f"<b>Cluster:</b> {arch_info['name']}<br>"
                f"<b>Field:</b> {field_str}<br>"
                f"<b>Goal:</b> {goal_str}<br>"
                f"<b>━━━ Self Ratings ━━━</b><br>"
                f"Attr: {row.get('attr_self', 'N/A'):.0f} | "
                f"Fun: {row.get('fun_self', 'N/A'):.0f} | "
                f"Intel: {row.get('intel_self', 'N/A'):.0f}<br>"
                f"Sinc: {row.get('sinc_self', 'N/A'):.0f} | "
                f"Amb: {row.get('amb_self', 'N/A'):.0f}<br>"
                f"<b>━━━ Behavior ━━━</b><br>"
                f"Go Out: {row.get('go_out', 'N/A'):.0f} | "
                f"Date Freq: {row.get('date_freq', 'N/A'):.0f}"
            )
            hover_texts.append(hover_text)
            custom_data.append(int(row['iid']))

        fig.add_trace(go.Scatter(
            x=cluster_data['extraversion'],
            y=cluster_data['rationality'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=arch_info['color'],
                symbol=symbols,
                opacity=0.75,
                line=dict(width=1, color='white')
            ),
            name=f"{arch_info['code']}: {arch_info['name']}",
            text=hover_texts,
            customdata=custom_data,
            hovertemplate='%{text}<extra></extra>'
        ))

    # Highlight selected participant
    if highlight_id is not None and highlight_id in filtered_df['iid'].values:
        highlight_row = filtered_df[filtered_df['iid'] == highlight_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[highlight_row['extraversion']],
            y=[highlight_row['rationality']],
            mode='markers',
            marker=dict(
                size=25,
                color='gold',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name=f"Selected: ID {highlight_id}",
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="2D Persona Projection Map",
            x=0.5,
            font=dict(size=20, family="Times New Roman")
        ),
        xaxis=dict(
            title=dict(text="← Introverted | Extraversion | Extroverted →", font=dict(size=13)),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(100,100,100,0.5)',
            range=[-3.5, 3.5],
            dtick=1
        ),
        yaxis=dict(
            title=dict(text="← Emotional | Rationality | Rational →", font=dict(size=13)),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(100,100,100,0.5)',
            range=[-3.5, 3.5],
            dtick=1
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02,
            font=dict(size=10)
        ),
        height=650,
        margin=dict(l=60, r=150, t=60, b=60),
        **ACADEMIC_LAYOUT
    )

    return fig, filtered_df

def create_profile_card(row, arch_info):
    """Generate HTML for participant profile card"""
    gender_str = 'Female' if row['gender'] == 0 else 'Male'
    gender_icon = '👩' if row['gender'] == 0 else '👨'
    goal_str = GOAL_MAP.get(row.get('goal', 0), 'Unknown')
    field_str = FIELD_MAP.get(row.get('field_cd', 0), 'Unknown')

    # Create trait bars
    def make_bar(value, max_val=10):
        if pd.isna(value):
            return "N/A"
        pct = min(100, (value / max_val) * 100)
        return f'<div style="background:#eee;border-radius:3px;height:8px;width:100px;display:inline-block;"><div style="background:{arch_info["color"]};height:8px;width:{pct}%;border-radius:3px;"></div></div> {value:.0f}'

    html = f"""
    <div style="background:white;padding:20px;border-radius:12px;box-shadow:0 4px 15px rgba(0,0,0,0.1);border-left:5px solid {arch_info['color']};">
        <div style="display:flex;align-items:center;margin-bottom:15px;">
            <span style="font-size:40px;margin-right:15px;">{gender_icon}</span>
            <div>
                <h3 style="margin:0;color:{arch_info['color']};">Participant #{int(row['iid'])}</h3>
                <span style="background:{arch_info['color']}22;color:{arch_info['color']};padding:2px 8px;border-radius:10px;font-size:12px;">{arch_info['name']}</span>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:15px;">
            <div><b>Age:</b> {row['age']:.0f}</div>
            <div><b>Gender:</b> {gender_str}</div>
            <div><b>Field:</b> {field_str}</div>
            <div><b>Goal:</b> {goal_str}</div>
        </div>

        <div style="border-top:1px solid #eee;padding-top:15px;">
            <h4 style="margin:0 0 10px 0;color:#555;">Self-Rated Traits</h4>
            <div style="margin:5px 0;"><b>Attractiveness:</b> {make_bar(row.get('attr_self'))}</div>
            <div style="margin:5px 0;"><b>Fun:</b> {make_bar(row.get('fun_self'))}</div>
            <div style="margin:5px 0;"><b>Intelligence:</b> {make_bar(row.get('intel_self'))}</div>
            <div style="margin:5px 0;"><b>Sincerity:</b> {make_bar(row.get('sinc_self'))}</div>
            <div style="margin:5px 0;"><b>Ambition:</b> {make_bar(row.get('amb_self'))}</div>
        </div>

        <div style="border-top:1px solid #eee;padding-top:15px;margin-top:15px;">
            <h4 style="margin:0 0 10px 0;color:#555;">Social Behavior</h4>
            <div style="margin:5px 0;"><b>Go Out Freq:</b> {make_bar(row.get('go_out'), 7)}</div>
            <div style="margin:5px 0;"><b>Dating Freq:</b> {make_bar(row.get('date_freq'), 7)}</div>
        </div>

        <div style="border-top:1px solid #eee;padding-top:15px;margin-top:15px;">
            <h4 style="margin:0 0 10px 0;color:#555;">Position Scores</h4>
            <div><b>Extraversion:</b> {row['extraversion']:.2f}</div>
            <div><b>Rationality:</b> {row['rationality']:.2f}</div>
        </div>
    </div>
    """
    return html

def create_aggregate_stats(df, arch_mapping):
    """Create aggregate statistics for selected participants"""
    if len(df) == 0:
        return None

    stats = {
        'count': len(df),
        'avg_age': df['age'].mean(),
        'male_pct': (df['gender'] == 1).mean() * 100,
        'avg_extraversion': df['extraversion'].mean(),
        'avg_rationality': df['rationality'].mean(),
        'avg_attr': df['attr_self'].mean(),
        'avg_fun': df['fun_self'].mean(),
        'avg_intel': df['intel_self'].mean(),
        'avg_sinc': df['sinc_self'].mean(),
        'avg_amb': df['amb_self'].mean(),
    }

    # Cluster distribution
    cluster_dist = df['cluster'].value_counts().to_dict()
    stats['cluster_dist'] = cluster_dist

    return stats

# ==================== Load and Process Data ====================
raw_df = load_data()
participant_df = preprocess_for_projection(raw_df)
participant_df = compute_custom_axes(participant_df)

# Perform clustering
labels, silhouette_score_val, kmeans = perform_clustering(participant_df, n_clusters=5)
participant_df['cluster'] = labels

# ==================== Sidebar Controls ====================
st.sidebar.header("🎛️ Filters")

# Gender filter
gender_options = {'Female': 0, 'Male': 1}
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=list(gender_options.keys()),
    default=list(gender_options.keys())
)
selected_gender_values = [gender_options[g] for g in selected_genders]

# Age range filter
valid_ages = participant_df['age'].dropna()
if len(valid_ages) > 0:
    min_age = int(valid_ages.min())
    max_age = int(valid_ages.max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
else:
    age_range = None

# Wave filter
if 'wave' in participant_df.columns:
    waves = sorted(participant_df['wave'].dropna().unique().astype(int))
    selected_waves = st.sidebar.multiselect("Event Waves", options=waves, default=waves)
else:
    selected_waves = None

# Cluster filter
cluster_options = ["All Clusters"]
cluster_map = {}
for cluster_id, arch_info in CLUSTER_ARCHETYPES.items():
    option_text = f"{arch_info['code']}: {arch_info['name']}"
    cluster_options.append(option_text)
    cluster_map[option_text] = cluster_id

selected_cluster_option = st.sidebar.selectbox("Cluster Type", options=cluster_options)
selected_cluster = None
if selected_cluster_option != "All Clusters":
    selected_cluster = cluster_map[selected_cluster_option]

# Goal filter
if 'goal' in participant_df.columns:
    available_goals = participant_df['goal'].dropna().unique()
    goal_options = {GOAL_MAP.get(int(g), f"Goal {int(g)}"): int(g) for g in available_goals if g in GOAL_MAP}
    selected_goals_text = st.sidebar.multiselect("Dating Goals", options=list(goal_options.keys()), default=list(goal_options.keys()))
    selected_goals = [goal_options[g] for g in selected_goals_text]
else:
    selected_goals = None

# ==================== Header ====================
st.title("🌌 2D Persona Projection Map")
st.markdown("""
**Interpretable Personality Space** — Each participant projected onto two meaningful psychological dimensions.

This visualization uses **interpretable axes** (not black-box dimensionality reduction):
- **X-axis (Extraversion):** `z(fun) + z(go_out) + z(date_freq)` — Higher = more social/outgoing
- **Y-axis (Rationality):** `z(intel) + z(amb) - z(sinc)` — Higher = more goal-oriented/analytical
""")

# ==================== Summary Stats ====================
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

valid_participants = participant_df.dropna(subset=['extraversion', 'rationality'])
with col1:
    st.metric("Total Participants", len(valid_participants))
with col2:
    st.metric("Clusters", len(CLUSTER_ARCHETYPES))
with col3:
    st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
with col4:
    male_pct = (valid_participants['gender'] == 1).mean() * 100
    st.metric("Male %", f"{male_pct:.1f}%")
with col5:
    avg_age = valid_participants['age'].mean()
    st.metric("Avg Age", f"{avg_age:.1f}")

st.markdown("---")

# ==================== Main Layout ====================
st.subheader("🗺️ Persona Projection Space")

# Create and display the plot
fig, filtered_df = create_projection_plot(
    participant_df,
    selected_gender=selected_gender_values if selected_genders else None,
    age_range=age_range,
    selected_wave=selected_waves,
    selected_cluster=selected_cluster,
    selected_goal=selected_goals,
    highlight_id=None
)

# Display plot with selection capability
st.plotly_chart(fig, use_container_width=True, key="main_scatter")

# Display filtered count
st.caption(f"Showing {len(filtered_df)} participants after filters applied")

# ==================== Selection Stats ====================
st.markdown("---")
st.subheader("📈 Selection Stats")

if len(filtered_df) > 0:
    stats = create_aggregate_stats(filtered_df, CLUSTER_ARCHETYPES)
    if stats:
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Selected Count", stats['count'])
        with stat_col2:
            st.metric("Avg Age", f"{stats['avg_age']:.1f}")
        with stat_col3:
            st.metric("Male %", f"{stats['male_pct']:.1f}%")

        # Cluster distribution
        st.markdown("**Cluster Distribution:**")
        dist_cols = st.columns(len(stats['cluster_dist']))
        for i, (cluster_id, count) in enumerate(sorted(stats['cluster_dist'].items())):
            if pd.notna(cluster_id):
                arch = CLUSTER_ARCHETYPES.get(int(cluster_id), {})
                pct = count / stats['count'] * 100
                with dist_cols[i]:
                    st.markdown(f"**{arch.get('code', '?')}:** {count} ({pct:.1f}%)")

# ==================== Visual Encoding Legend ====================
st.markdown("---")
st.subheader("📖 Visual Encoding Guide")

leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)

with leg_col1:
    st.markdown("**Symbol Shapes**")
    st.markdown("● Circle = Female")
    st.markdown("▲ Triangle = Male")

with leg_col2:
    st.markdown("**Point Size**")
    st.markdown("Larger = Higher self-rated attractiveness")

with leg_col3:
    st.markdown("**Point Color**")
    st.markdown("Based on K-Means cluster assignment")

with leg_col4:
    st.markdown("**Background Zones**")
    st.markdown("Quadrants indicate personality combinations")

# ==================== Cluster Archetypes ====================
st.markdown("---")
st.subheader("🧬 Cluster Archetypes")

cluster_cols = st.columns(len(CLUSTER_ARCHETYPES))
for i, (cluster_id, arch_info) in enumerate(CLUSTER_ARCHETYPES.items()):
    cluster_count = (participant_df['cluster'] == cluster_id).sum()
    total = len(participant_df.dropna(subset=['cluster']))
    pct = cluster_count / total * 100 if total > 0 else 0

    with cluster_cols[i]:
        st.markdown(f"""
        <div style="background:{arch_info['color']}15;padding:12px;border-radius:10px;border-left:4px solid {arch_info['color']};height:100%;">
            <b style="color:{arch_info['color']};font-size:14px;">{arch_info['code']}: {arch_info['name']}</b><br>
            <small><b>n={cluster_count}</b> ({pct:.1f}%)</small><br>
            <small style="color:#666;">{arch_info['description']}</small>
        </div>
        """, unsafe_allow_html=True)

# ==================== Footer ====================
st.markdown("---")
st.caption("""
**Methodology:**
- **Projection Method:** Custom interpretable axes (not PCA/t-SNE/UMAP black-box reduction)
- **X-Axis (Extraversion):** Standardized sum of fun + go_out_frequency + dating_frequency
- **Y-Axis (Rationality):** Standardized combination of intelligence + ambition - sincerity
- **Clustering:** K-Means (k=5) on the 2D projection coordinates
- **Data:** Speed Dating Experiment Dataset | Self-assessed traits (attr3_1, fun3_1, etc.)

**Note:** All scores standardized to [-3, 3] range. This analysis describes participant profiles only, not interaction outcomes.
""")
