import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# ==================== Page Config ====================
st.set_page_config(
    page_title="Participants Portrait Map",
    page_icon="🌐",
    layout="wide"
)

# ==================== Academic Style ====================
ACADEMIC_FONT = dict(family="Times New Roman, serif", size=12)
ACADEMIC_LAYOUT = dict(
    font=ACADEMIC_FONT,
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# ==================== Data Loading ====================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Speed Dating Data.csv", encoding='latin-1')
    return df

@st.cache_data
def prepare_participant_data(df):
    """Prepare participant-level data with all relevant features"""
    # Core columns
    demo_cols = ['iid', 'gender', 'age', 'race', 'field_cd', 'career_c',
                 'date', 'go_out', 'goal', 'imprace', 'imprelig', 'wave']

    # Self-perception (6 dimensions)
    self_cols = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']

    # Partner preferences
    pref_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']

    # Lifestyle
    lifestyle_cols = ['sports', 'tvsports', 'exercise', 'dining', 'museums',
                      'art', 'hiking', 'gaming', 'clubbing', 'reading',
                      'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']

    all_cols = demo_cols + self_cols + pref_cols + lifestyle_cols
    available_cols = [c for c in all_cols if c in df.columns]

    # Aggregate to participant level
    participant_df = df.groupby('iid').first().reset_index()[available_cols]

    # Clean data
    numeric_cols = ['age', 'date', 'go_out', 'imprace', 'imprelig'] + \
                   [c for c in self_cols if c in participant_df.columns] + \
                   [c for c in pref_cols if c in participant_df.columns] + \
                   [c for c in lifestyle_cols if c in participant_df.columns]

    for col in numeric_cols:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            participant_df[col] = participant_df[col].fillna(participant_df[col].median())

    return participant_df, self_cols, pref_cols

@st.cache_data
def perform_clustering(df, feature_cols, n_clusters):
    """Perform KMeans clustering on specified features"""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0

    return labels, score

# ==================== Star Plot Functions ====================
def create_mini_radar_svg(values, size=30, color='#4ECDC4'):
    """Create a mini radar chart as SVG path"""
    n = len(values)
    if n == 0:
        return ""

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Close the polygon
    values = np.concatenate([values, [values[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    # Convert to cartesian
    x = size/2 + (values * size/2 * 0.8) * np.cos(angles - np.pi/2)
    y = size/2 + (values * size/2 * 0.8) * np.sin(angles - np.pi/2)

    # Create SVG path
    path = f"M {x[0]},{y[0]} "
    for i in range(1, len(x)):
        path += f"L {x[i]},{y[i]} "
    path += "Z"

    return path

def create_starfield_figure(df, dim_cols, sort_by, n_cols=20, selected_cluster=None):
    """
    Create the 6D Personality Starfield visualization.
    Each participant is a mini radar chart arranged in a grid.
    """
    # Filter data
    plot_df = df.copy()
    if selected_cluster is not None and 'cluster' in plot_df.columns:
        plot_df = plot_df[plot_df['cluster'] == selected_cluster]

    # Get available dimension columns
    available_dims = [c for c in dim_cols if c in plot_df.columns]
    if len(available_dims) == 0:
        return None

    # Normalize dimensions to 0-1
    scaler = MinMaxScaler()
    dim_values = scaler.fit_transform(plot_df[available_dims])
    plot_df[available_dims] = dim_values

    # Sort
    if sort_by == 'Cluster':
        plot_df = plot_df.sort_values(['cluster', 'iid'])
    elif sort_by == 'Average Score':
        plot_df['avg_score'] = plot_df[available_dims].mean(axis=1)
        plot_df = plot_df.sort_values('avg_score', ascending=False)
    elif sort_by in available_dims:
        plot_df = plot_df.sort_values(sort_by, ascending=False)
    elif sort_by == 'Gender':
        plot_df = plot_df.sort_values(['gender', 'age'])
    elif sort_by == 'Age':
        plot_df = plot_df.sort_values('age')

    # Limit for performance
    max_display = 200
    if len(plot_df) > max_display:
        plot_df = plot_df.head(max_display)

    n_participants = len(plot_df)
    n_rows = (n_participants + n_cols - 1) // n_cols

    # Create figure
    fig = go.Figure()

    # Cell size
    cell_size = 40
    padding = 5

    # Dimension labels
    dim_labels = {
        'attr3_1': 'Attr', 'sinc3_1': 'Sinc', 'intel3_1': 'Intel',
        'fun3_1': 'Fun', 'amb3_1': 'Amb',
        'attr1_1': 'Attr', 'sinc1_1': 'Sinc', 'intel1_1': 'Intel',
        'fun1_1': 'Fun', 'amb1_1': 'Amb', 'shar1_1': 'Shar'
    }

    # Cluster colors
    cluster_colors = px.colors.qualitative.Set2

    # Create each mini radar
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        col_idx = idx % n_cols
        row_idx = idx // n_cols

        x_center = col_idx * (cell_size + padding) + cell_size / 2
        y_center = (n_rows - 1 - row_idx) * (cell_size + padding) + cell_size / 2

        # Get values for this participant
        values = row[available_dims].values.astype(float)

        # Determine color by cluster
        if 'cluster' in row:
            color = cluster_colors[int(row['cluster']) % len(cluster_colors)]
        else:
            color = '#4ECDC4'

        # Create radar polygon
        n_dims = len(values)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)

        # Radar vertices
        radius = cell_size / 2 * 0.8
        x_points = x_center + values * radius * np.cos(angles - np.pi/2)
        y_points = y_center + values * radius * np.sin(angles - np.pi/2)

        # Close polygon
        x_points = np.append(x_points, x_points[0])
        y_points = np.append(y_points, y_points[0])

        # Build hover text
        hover_parts = [f"<b>ID: {int(row['iid'])}</b>"]
        hover_parts.append(f"Gender: {'Male' if row.get('gender', 0) == 1 else 'Female'}")
        hover_parts.append(f"Age: {row.get('age', 'N/A'):.0f}")
        if 'cluster' in row:
            hover_parts.append(f"Cluster: Type {chr(65 + int(row['cluster']))}")
        hover_parts.append("<br><b>Dimensions:</b>")
        for dim in available_dims:
            label = dim_labels.get(dim, dim)
            val = row[dim]
            hover_parts.append(f"  {label}: {val:.2f}")

        hover_text = "<br>".join(hover_parts)

        # Add filled polygon
        fig.add_trace(go.Scatter(
            x=x_points, y=y_points,
            fill='toself',
            fillcolor=color,
            opacity=0.6,
            line=dict(color=color, width=1),
            mode='lines',
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))

        # Add outer circle (reference)
        theta = np.linspace(0, 2*np.pi, 30)
        circle_x = x_center + radius * np.cos(theta)
        circle_y = y_center + radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='lightgrey', width=0.5),
            hoverinfo='skip',
            showlegend=False
        ))

    # Update layout
    total_width = n_cols * (cell_size + padding)
    total_height = n_rows * (cell_size + padding)

    fig.update_layout(
        width=min(1200, total_width + 100),
        height=min(800, total_height + 100),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-20, total_width + 20]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-20, total_height + 20],
            scaleanchor='x', scaleratio=1
        ),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(
            text=f"Personality Starfield ({len(plot_df)} participants)",
            x=0.5,
            font=dict(color='white', size=16)
        )
    )

    return fig

# ==================== Load Data ====================
raw_df = load_data()
participant_df, self_cols, pref_cols = prepare_participant_data(raw_df)

# ==================== Sidebar ====================
st.sidebar.header("🎛️ Controls")

# Dimension selection
dim_choice = st.sidebar.radio(
    "Dimension Type",
    options=["Self-Perception", "Partner Preference"],
    index=0
)

if dim_choice == "Self-Perception":
    dim_cols = self_cols
    dim_title = "Self-Perception Dimensions"
else:
    dim_cols = pref_cols
    dim_title = "Partner Preference Dimensions"

# Number of clusters
n_clusters = st.sidebar.slider("Number of Clusters", 3, 6, 4)

# Perform clustering
available_dims = [c for c in dim_cols if c in participant_df.columns]
cluster_labels, silhouette = perform_clustering(participant_df, available_dims, n_clusters)
participant_df['cluster'] = cluster_labels

# Sort options
sort_options = ['Cluster', 'Average Score', 'Gender', 'Age'] + available_dims
sort_by = st.sidebar.selectbox("Sort By", options=sort_options, index=0)

# Cluster filter
cluster_filter = st.sidebar.selectbox(
    "Filter Cluster",
    options=["All"] + [f"Type {chr(65+i)}" for i in range(n_clusters)],
    index=0
)
selected_cluster = None if cluster_filter == "All" else ord(cluster_filter[-1]) - 65

# Wave filter
waves = sorted(participant_df['wave'].dropna().unique())
selected_waves = st.sidebar.multiselect("Filter by Wave", options=waves, default=waves)

# Filter by wave
wave_mask = participant_df['wave'].isin(selected_waves)
df_filtered = participant_df[wave_mask].copy()

# ==================== Header ====================
st.title("🌐 Participants Portrait Map")
st.markdown("""
**6D Personality Starfield Visualization**

Each "star" represents a participant's multi-dimensional profile as a mini radar chart.
The shape reveals their personality structure at a glance.
""")

# ==================== Summary Stats ====================
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Participants", len(df_filtered))
with col2:
    male_pct = (df_filtered['gender'] == 1).mean() * 100
    st.metric("Male Ratio", f"{male_pct:.1f}%")
with col3:
    st.metric("Avg Age", f"{df_filtered['age'].mean():.1f}")
with col4:
    st.metric("Clusters", n_clusters)
with col5:
    st.metric("Silhouette", f"{silhouette:.3f}")

st.markdown("---")

# ==================== Starfield Visualization ====================
st.subheader(f"✨ {dim_title} Starfield")

fig_starfield = create_starfield_figure(
    df_filtered,
    dim_cols=available_dims,
    sort_by=sort_by,
    n_cols=15,
    selected_cluster=selected_cluster
)

if fig_starfield:
    st.plotly_chart(fig_starfield)
else:
    st.warning("No data available for visualization.")

# Legend
st.markdown("#### Legend")
legend_cols = st.columns(n_clusters)
cluster_colors = px.colors.qualitative.Set2
for i, col in enumerate(legend_cols):
    with col:
        color = cluster_colors[i % len(cluster_colors)]
        st.markdown(f"""
        <div style="display:flex;align-items:center;">
            <div style="width:20px;height:20px;background:{color};border-radius:50%;margin-right:10px;"></div>
            <span>Type {chr(65+i)}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==================== Cluster Profiles ====================
st.subheader("🔍 Cluster Profile Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Radar Comparison", "📈 Demographics", "📋 Summary Table"])

with tab1:
    # Full-size radar comparison
    dim_labels = {
        'attr3_1': 'Attractiveness', 'sinc3_1': 'Sincerity', 'intel3_1': 'Intelligence',
        'fun3_1': 'Fun', 'amb3_1': 'Ambition',
        'attr1_1': 'Attractiveness', 'sinc1_1': 'Sincerity', 'intel1_1': 'Intelligence',
        'fun1_1': 'Fun', 'amb1_1': 'Ambition', 'shar1_1': 'Shared Interests'
    }

    cluster_means = df_filtered.groupby('cluster')[available_dims].mean()
    scaler = MinMaxScaler()
    cluster_norm = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )

    fig_radar = go.Figure()

    for cluster_id in range(n_clusters):
        if cluster_id in cluster_norm.index:
            values = cluster_norm.loc[cluster_id].values.tolist()
            values.append(values[0])
            categories = [dim_labels.get(c, c) for c in available_dims]
            categories.append(categories[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories,
                fill='toself',
                name=f'Type {chr(65+cluster_id)}',
                opacity=0.6,
                line=dict(color=cluster_colors[cluster_id % len(cluster_colors)])
            ))

    fig_radar.update_layout(
        title=dict(text=f"{dim_title} by Cluster", x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='white'
        ),
        showlegend=True,
        height=500,
        **ACADEMIC_LAYOUT
    )

    st.plotly_chart(fig_radar)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        # Age by cluster
        fig_age = go.Figure()
        for cluster_id in range(n_clusters):
            cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
            fig_age.add_trace(go.Box(
                y=cluster_data['age'],
                name=f'Type {chr(65+cluster_id)}',
                marker_color=cluster_colors[cluster_id % len(cluster_colors)]
            ))

        fig_age.update_layout(
            title=dict(text="Age Distribution by Cluster", x=0.5),
            yaxis_title="Age",
            showlegend=False,
            height=350,
            **ACADEMIC_LAYOUT
        )
        st.plotly_chart(fig_age)

    with col_b:
        # Gender by cluster
        gender_data = df_filtered.groupby(['cluster', 'gender']).size().unstack(fill_value=0)
        gender_pct = gender_data.div(gender_data.sum(axis=1), axis=0) * 100

        fig_gender = go.Figure()
        for gender, label in [(0, 'Female'), (1, 'Male')]:
            if gender in gender_pct.columns:
                fig_gender.add_trace(go.Bar(
                    x=[f'Type {chr(65+i)}' for i in gender_pct.index],
                    y=gender_pct[gender],
                    name=label,
                    marker_color='#FF6B6B' if gender == 0 else '#2E86AB'
                ))

        fig_gender.update_layout(
            title=dict(text="Gender Composition by Cluster", x=0.5),
            yaxis_title="Percentage (%)",
            barmode='stack',
            height=350,
            **ACADEMIC_LAYOUT
        )
        st.plotly_chart(fig_gender)

    col_c, col_d = st.columns(2)

    with col_c:
        # Social activity
        fig_social = go.Figure()
        for cluster_id in range(n_clusters):
            cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
            fig_social.add_trace(go.Box(
                y=cluster_data['go_out'],
                name=f'Type {chr(65+cluster_id)}',
                marker_color=cluster_colors[cluster_id % len(cluster_colors)]
            ))

        fig_social.update_layout(
            title=dict(text="Social Activity (Go Out) by Cluster", x=0.5),
            yaxis_title="Frequency",
            showlegend=False,
            height=350,
            **ACADEMIC_LAYOUT
        )
        st.plotly_chart(fig_social)

    with col_d:
        # Dating frequency
        fig_date = go.Figure()
        for cluster_id in range(n_clusters):
            cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
            fig_date.add_trace(go.Box(
                y=cluster_data['date'],
                name=f'Type {chr(65+cluster_id)}',
                marker_color=cluster_colors[cluster_id % len(cluster_colors)]
            ))

        fig_date.update_layout(
            title=dict(text="Dating Frequency by Cluster", x=0.5),
            yaxis_title="Frequency",
            showlegend=False,
            height=350,
            **ACADEMIC_LAYOUT
        )
        st.plotly_chart(fig_date)

with tab3:
    # Summary table
    summary_cols = ['age', 'gender', 'date', 'go_out'] + available_dims

    cluster_summary = df_filtered.groupby('cluster').agg({
        'iid': 'count',
        'age': 'mean',
        'gender': 'mean',
        'date': 'mean',
        'go_out': 'mean',
        **{col: 'mean' for col in available_dims}
    }).round(2)

    col_names = ['Count', 'Avg Age', 'Male %', 'Dating Freq', 'Go Out Freq'] + \
                [dim_labels.get(c, c) for c in available_dims]
    cluster_summary.columns = col_names
    cluster_summary['Male %'] = (cluster_summary['Male %'] * 100).round(1)
    cluster_summary.index = [f'Type {chr(65+i)}' for i in cluster_summary.index]

    st.dataframe(cluster_summary)

    # Interpretations
    st.markdown("#### Cluster Interpretations")

    for cluster_id in range(n_clusters):
        if cluster_id in cluster_means.index:
            row = cluster_summary.loc[f'Type {chr(65+cluster_id)}']
            size = int(row['Count'])
            pct = size / len(df_filtered) * 100

            # Find dominant trait
            dim_values = cluster_means.loc[cluster_id]
            top_dim = dim_values.idxmax()
            top_dim_name = dim_labels.get(top_dim, top_dim)

            # Gender note
            male_ratio = row['Male %']
            if male_ratio > 60:
                gender_note = "predominantly male"
            elif male_ratio < 40:
                gender_note = "predominantly female"
            else:
                gender_note = "gender balanced"

            # Social note
            social_median = cluster_summary['Go Out Freq'].median()
            social_note = "socially active" if row['Go Out Freq'] > social_median else "socially reserved"

            st.markdown(f"""
            **Type {chr(65+cluster_id)}** ({size} participants, {pct:.1f}%) — {gender_note}
            - Primary trait: **{top_dim_name}**
            - Characteristics: {social_note}
            - Avg age: {row['Avg Age']:.1f}
            """)

# ==================== Footer ====================
st.markdown("---")
st.caption("""
**Visualization:** 6D Personality Starfield — Each mini radar chart represents a participant's multi-dimensional profile.
**Data Source:** Speed Dating Experiment Dataset
**Method:** KMeans clustering on self-perception or preference dimensions.
""")
