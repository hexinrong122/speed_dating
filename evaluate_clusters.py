"""
Clustering Evaluation Script
Evaluate clustering performance for K=2 to K=10 using multiple metrics
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def prepare_data():
    """Prepare data"""
    # Load data
    df = pd.read_csv('data/Speed Dating Data.csv', encoding='latin-1')

    # Select columns
    demo_cols = ['iid', 'gender', 'age', 'race', 'field_cd', 'career_c',
                 'date', 'go_out', 'goal', 'imprace', 'imprelig', 'wave']
    pref_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    self_cols = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']

    all_cols = demo_cols + pref_cols + self_cols
    available_cols = [c for c in all_cols if c in df.columns]

    # Aggregate to participant level
    participant_df = df.groupby('iid').first().reset_index()[available_cols]

    # Data cleaning
    for col in ['age', 'date', 'go_out', 'imprace', 'imprelig'] + pref_cols + self_cols:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            participant_df[col] = participant_df[col].fillna(participant_df[col].median())

    # One-hot encoding
    for col in ['race', 'field_cd', 'career_c', 'goal']:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            dummies = pd.get_dummies(participant_df[col], prefix=col, dummy_na=False)
            participant_df = pd.concat([participant_df, dummies], axis=1)

    # Build feature matrix
    feature_cols = []
    for col in ['age', 'gender', 'date', 'go_out', 'imprace', 'imprelig']:
        if col in participant_df.columns:
            feature_cols.append(col)
    for col in pref_cols + self_cols:
        if col in participant_df.columns:
            feature_cols.append(col)
    for col in participant_df.columns:
        if any(col.startswith(cat + '_') for cat in ['race', 'field_cd', 'career_c', 'goal']):
            feature_cols.append(col)

    X = participant_df[feature_cols].copy()
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

def evaluate_clusters(X_scaled, k_range=range(2, 11)):
    """Evaluate clustering performance for different K values"""
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'ch_score': calinski_harabasz_score(X_scaled, labels),
            'db_score': davies_bouldin_score(X_scaled, labels)
        })

    return pd.DataFrame(results)

def plot_evaluation(results_df):
    """Plot evaluation charts with academic style"""
    # Create subplots with academic style
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Elbow Method (Inertia)',
            'Silhouette Score',
            'Calinski-Harabasz Index',
            'Davies-Bouldin Index'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Inertia (Elbow)
    fig.add_trace(
        go.Scatter(x=results_df['k'], y=results_df['inertia'],
                   mode='lines+markers', name='Inertia',
                   marker=dict(size=8, color='blue'), 
                   line=dict(width=2, color='blue'),
                   hovertemplate='K: %{x}<br>Inertia: %{y:.2f}<extra></extra>'),
        row=1, col=1
    )

    # Silhouette
    fig.add_trace(
        go.Scatter(x=results_df['k'], y=results_df['silhouette'],
                   mode='lines+markers', name='Silhouette',
                   marker=dict(size=8, color='green'), 
                   line=dict(width=2, color='green'),
                   hovertemplate='K: %{x}<br>Silhouette: %{y:.4f}<extra></extra>'),
        row=1, col=2
    )

    # Calinski-Harabasz
    fig.add_trace(
        go.Scatter(x=results_df['k'], y=results_df['ch_score'],
                   mode='lines+markers', name='CH Index',
                   marker=dict(size=8, color='orange'), 
                   line=dict(width=2, color='orange'),
                   hovertemplate='K: %{x}<br>CH Index: %{y:.2f}<extra></extra>'),
        row=2, col=1
    )

    # Davies-Bouldin
    fig.add_trace(
        go.Scatter(x=results_df['k'], y=results_df['db_score'],
                   mode='lines+markers', name='DB Index',
                   marker=dict(size=8, color='red'), 
                   line=dict(width=2, color='red'),
                   hovertemplate='K: %{x}<br>DB Index: %{y:.4f}<extra></extra>'),
        row=2, col=2
    )

    # Mark optimal points
    best_sil = results_df.loc[results_df['silhouette'].idxmax()]
    best_ch = results_df.loc[results_df['ch_score'].idxmax()]
    best_db = results_df.loc[results_df['db_score'].idxmin()]

    fig.add_annotation(x=best_sil['k'], y=best_sil['silhouette'],
                       text=f"Optimal: K={int(best_sil['k'])}", 
                       showarrow=True, arrowhead=1, ax=20, ay=-40,
                       row=1, col=2)
    
    fig.add_annotation(x=best_ch['k'], y=best_ch['ch_score'],
                       text=f"Optimal: K={int(best_ch['k'])}", 
                       showarrow=True, arrowhead=1, ax=20, ay=-40,
                       row=2, col=1)
    
    fig.add_annotation(x=best_db['k'], y=best_db['db_score'],
                       text=f"Optimal: K={int(best_db['k'])}", 
                       showarrow=True, arrowhead=1, ax=-20, ay=-40,
                       row=2, col=2)

    # Update layout for academic style
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Cluster Number Evaluation (K=2 to K=10)",
        title_x=0.5,
        font=dict(family="Times New Roman, serif", size=12),
        plot_bgcolor='white'
    )

    # Update axes for academic style
    fig.update_xaxes(title_text="Number of Clusters (K)", dtick=1, 
                     showline=True, linewidth=1, linecolor='black',
                     mirror=True, gridcolor='lightgrey')
    
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                     mirror=True, gridcolor='lightgrey')

    return fig

if __name__ == "__main__":
    print("=" * 65)
    print("聚类数量评估")
    print("=" * 65)
    print()

    # Prepare data
    print("正在准备数据...")
    X_scaled = prepare_data()
    print(f"样本数: {len(X_scaled)}, 特征数: {X_scaled.shape[1]}")
    print()

    # Evaluate
    print("正在评估 K=2 到 K=10...")
    results_df = evaluate_clusters(X_scaled)

    # Print results table
    print()
    print("=" * 70)
    print(f"{'K':>3} | {'Inertia':>12} | {'Silhouette':>10} | {'CH Index':>12} | {'DB Index':>10}")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"{int(row['k']):>3} | {row['inertia']:>12.1f} | {row['silhouette']:>10.4f} | {row['ch_score']:>12.1f} | {row['db_score']:>10.4f}")
    print("=" * 70)
    print()

    # Recommendation
    best_sil_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
    best_ch_k = results_df.loc[results_df['ch_score'].idxmax(), 'k']
    best_db_k = results_df.loc[results_df['db_score'].idxmin(), 'k']

    print("指标说明:")
    print("  - Inertia: 越小越好 (肘部法则找拐点)")
    print("  - Silhouette: 越大越好 (-1到1，越接近1越好)")
    print("  - CH Index (Calinski-Harabasz): 越大越好")
    print("  - DB Index (Davies-Bouldin): 越小越好")
    print()
    print("最佳K推荐:")
    print(f"  - 基于 Silhouette Score: K = {int(best_sil_k)}")
    print(f"  - 基于 CH Index: K = {int(best_ch_k)}")
    print(f"  - 基于 DB Index: K = {int(best_db_k)}")
    print()

    # Overall recommendation
    votes = [best_sil_k, best_ch_k, best_db_k]
    from collections import Counter
    vote_counts = Counter(votes)
    recommended_k = vote_counts.most_common(1)[0][0]
    print(f">>> 综合推荐: K = {int(recommended_k)} <<<")
    print()

    # Generate plots
    print("正在生成评估图表...")
    fig = plot_evaluation(results_df)
    fig.write_html("cluster_evaluation.html")
    print("图表已保存到: cluster_evaluation.html")
    fig.show()