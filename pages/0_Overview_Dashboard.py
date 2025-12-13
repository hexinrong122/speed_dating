import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit.components.v1 as components
import json

# ==================== Page Config ====================
st.set_page_config(
    page_title="Speed Dating Overview",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Style Constants ====================
BACKGROUND_COLOR = '#FFFFFF'
CARD_BG = '#FFFFFF'
MALE_COLOR = '#5f89d1'
FEMALE_COLOR = '#f2dada'
MATCH_COLOR = '#000000'
GRID_COLOR = '#a6c3d8'
TEXT_COLOR = '#000000'
DARK_BLUE = '#104a5b'
MEDIUM_BLUE = '#476d9e'
MEDIUM_PINK = '#e63b55'
DARK_RED = '#b22222'

# Cluster colors for persona projection
CLUSTER_COLORS = {
    0: "#3498DB", 1: "#9B59B6", 2: "#E74C3C", 3: "#2ECC71", 4: "#F39C12",
}
CLUSTER_NAMES = {
    0: "Social Extroverts", 1: "Rational Thinkers", 2: "Appearance-oriented",
    3: "Warm & Sincere", 4: "Quiet Observers",
}

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
    .stMetric { background: white; padding: 8px; border-radius: 6px; border: 1px solid #eee; }
    h1, h2, h3 { margin-top: 0 !important; padding-top: 0 !important; }
    .section-title { font-size: 13px; font-weight: bold; color: #333; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==================== Data Loading ====================
@st.cache_data
def load_data():
    return pd.read_csv("data/Speed Dating Data.csv", encoding='latin-1')

@st.cache_data
def process_match_data(df):
    cols = ['iid', 'pid', 'gender', 'wave', 'match', 'dec', 'dec_o', 'age',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
    available_cols = [c for c in cols if c in df.columns]
    df_clean = df[available_cols].copy()
    df_clean = df_clean.dropna(subset=['dec', 'dec_o', 'iid', 'pid'])
    df_clean['iid'] = df_clean['iid'].astype(int)
    df_clean['pid'] = df_clean['pid'].astype(int)
    df_clean['dec'] = df_clean['dec'].astype(int)
    df_clean['dec_o'] = df_clean['dec_o'].astype(int)
    df_clean['wave'] = df_clean['wave'].astype(int)
    df_clean['gender'] = df_clean['gender'].astype(int)
    df_clean['match'] = ((df_clean['dec'] == 1) & (df_clean['dec_o'] == 1)).astype(int)
    df_clean['uid'] = df_clean['wave'].astype(str) + '_' + df_clean['iid'].astype(str)
    df_clean['uid_partner'] = df_clean['wave'].astype(str) + '_' + df_clean['pid'].astype(str)
    df_clean['pair_key'] = df_clean.apply(lambda r: tuple(sorted([r['iid'], r['pid']])), axis=1)
    df_clean['pair_key_wave'] = df_clean['wave'].astype(str) + '_' + df_clean['pair_key'].astype(str)
    return df_clean

@st.cache_data
def build_network_data(df):
    yes_edges = df[df['dec'] == 1][['uid', 'uid_partner', 'gender', 'wave']].copy()
    yes_edges.columns = ['source', 'target', 'source_gender', 'wave']
    match_edges = df[df['match'] == 1][['uid', 'uid_partner', 'wave']].copy()
    match_edges.columns = ['source', 'target', 'wave']
    return yes_edges, match_edges

@st.cache_data
def process_persona_data(df):
    df_clean = df.dropna(subset=['iid', 'wave']).copy()
    participant_df = df_clean.groupby('iid').first().reset_index()
    column_mapping = {
        'fun3_1': 'fun_self', 'intel3_1': 'intel_self', 'sinc3_1': 'sinc_self',
        'amb3_1': 'amb_self', 'attr3_1': 'attr_self', 'go_out': 'go_out', 'date': 'date_freq',
    }
    for old_col, new_col in column_mapping.items():
        if old_col in participant_df.columns:
            participant_df[new_col] = pd.to_numeric(participant_df[old_col], errors='coerce')
    required_cols = ['fun_self', 'intel_self', 'sinc_self', 'amb_self', 'attr_self', 'go_out', 'date_freq']
    for col in required_cols:
        if col not in participant_df.columns:
            participant_df[col] = np.nan
    for col in required_cols + ['age']:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            participant_df[col] = participant_df[col].fillna(participant_df[col].median())
    if 'go_out' in participant_df.columns:
        participant_df['go_out_reversed'] = 8 - participant_df['go_out']
    return participant_df

def standardize_column(series):
    if series.std() == 0 or series.isna().all():
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / series.std()

@st.cache_data
def compute_persona_axes(participant_df):
    df = participant_df.copy()
    z_fun = standardize_column(df['fun_self'])
    z_go_out = standardize_column(df['go_out_reversed']) if 'go_out_reversed' in df.columns else 0
    z_date = standardize_column(df['date_freq'])
    z_intel = standardize_column(df['intel_self'])
    z_amb = standardize_column(df['amb_self'])
    z_sinc = standardize_column(df['sinc_self'])
    df['extraversion_raw'] = z_fun + z_go_out + z_date
    df['rationality_raw'] = z_intel + z_amb - z_sinc
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
    X = df[['extraversion', 'rationality']].dropna().values
    valid_indices = df[['extraversion', 'rationality']].dropna().index
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    full_labels = pd.Series(index=df.index, dtype=float)
    full_labels[valid_indices] = labels
    return full_labels.values

# ==================== Visualization Functions ====================
def create_sankey(both_yes, male_yes_female_no, female_yes_male_no, both_no):
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        textfont=dict(color='#333333', size=10, family='Arial'),
        node=dict(
            pad=15, thickness=20,
            line=dict(color='#666666', width=1),
            label=['All Interactions', 'Male Yes', 'Male No',
                   'Mutual Match', 'M Yes, F No', 'F Yes, M No', 'Both No'],
            color=['#f2dada', '#5f89d1', '#cccccc', '#e63b55', '#5f89d1', '#f2dada', '#666666'],
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 5, 6],
            value=[both_yes + male_yes_female_no, female_yes_male_no + both_no,
                   both_yes, male_yes_female_no, female_yes_male_no, both_no],
            color=['rgba(95,137,209,0.5)', 'rgba(204,204,204,0.5)', 'rgba(230,59,85,0.6)',
                   'rgba(95,137,209,0.4)', 'rgba(242,218,218,0.5)', 'rgba(102,102,102,0.5)']
        )
    ))
    fig.update_layout(
        height=200, margin=dict(l=5, r=5, t=25, b=5),
        paper_bgcolor='white', plot_bgcolor='white',
        title=dict(text="Decision Flow", font=dict(color='#333', size=12), x=0.5)
    )
    return fig

def create_persona_scatter(df):
    filtered_df = df.dropna(subset=['extraversion', 'rationality', 'cluster'])
    fig = go.Figure()
    # Background zones
    fig.add_shape(type="rect", x0=0, y0=0, x1=3.2, y1=3.2, line=dict(width=0), fillcolor="rgba(155,89,182,0.05)")
    fig.add_shape(type="rect", x0=-3.2, y0=0, x1=0, y1=3.2, line=dict(width=0), fillcolor="rgba(52,152,219,0.05)")
    fig.add_shape(type="rect", x0=-3.2, y0=-3.2, x1=0, y1=0, line=dict(width=0), fillcolor="rgba(46,204,113,0.05)")
    fig.add_shape(type="rect", x0=0, y0=-3.2, x1=3.2, y1=0, line=dict(width=0), fillcolor="rgba(243,156,18,0.05)")

    for cluster_id in sorted(filtered_df['cluster'].dropna().unique()):
        cluster_id = int(cluster_id)
        cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
        symbols = ['circle' if g == 0 else 'triangle-up' for g in cluster_data['gender']]
        sizes = [max(4, min(10, attr * 1.0)) if pd.notna(attr) else 6 for attr in cluster_data['attr_self']]
        fig.add_trace(go.Scatter(
            x=cluster_data['extraversion'], y=cluster_data['rationality'], mode='markers',
            marker=dict(size=sizes, color=CLUSTER_COLORS.get(cluster_id, '#888'), symbol=symbols, opacity=0.7, line=dict(width=0.5, color='white')),
            name=CLUSTER_NAMES.get(cluster_id, f'C{cluster_id}'),
            hovertemplate='Extraversion: %{x:.2f}<br>Rationality: %{y:.2f}<extra></extra>',
        ))
    fig.update_layout(
        xaxis=dict(title="Extraversion", showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinewidth=1, zerolinecolor='rgba(100,100,100,0.3)', range=[-3.5, 3.5], dtick=1, tickfont=dict(size=8)),
        yaxis=dict(title="Rationality", showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinewidth=1, zerolinecolor='rgba(100,100,100,0.3)', range=[-3.5, 3.5], dtick=1, tickfont=dict(size=8)),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='center', x=0.5, font=dict(size=8)),
        height=320, margin=dict(l=40, r=10, t=30, b=40),
        paper_bgcolor='white', plot_bgcolor='#fafafa',
    )
    return fig

def prepare_participant_profiles(df):
    self_cols = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
    available_cols = [c for c in self_cols if c in df.columns]
    if not available_cols:
        return None
    participant_df = df.groupby('uid').first().reset_index()
    scaler = MinMaxScaler()
    for col in available_cols:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            participant_df[col] = participant_df[col].fillna(participant_df[col].median())
    if len(participant_df) > 0 and all(c in participant_df.columns for c in available_cols):
        participant_df[available_cols] = scaler.fit_transform(participant_df[available_cols])
    return participant_df, available_cols

def create_cosmic_nebula_html(df, match_edges, yes_edges):
    result = prepare_participant_profiles(df)
    if result is None:
        return None
    participant_df, dim_cols = result
    match_dict = {}
    for _, row in match_edges.iterrows():
        src, tgt = row['source'], row['target']
        if src not in match_dict: match_dict[src] = []
        if tgt not in match_dict: match_dict[tgt] = []
        match_dict[src].append(tgt)
        match_dict[tgt].append(src)
    interaction_dict = {}
    for _, row in yes_edges.iterrows():
        src, tgt = row['source'], row['target']
        if src not in interaction_dict: interaction_dict[src] = {'said_yes': [], 'received_yes': []}
        interaction_dict[src]['said_yes'].append(tgt)
    for _, row in df[df['dec_o'] == 1].iterrows():
        uid, partner = row['uid'], row['uid_partner']
        if uid not in interaction_dict: interaction_dict[uid] = {'said_yes': [], 'received_yes': []}
        interaction_dict[uid]['received_yes'].append(partner)

    participants_data = []
    n_participants = len(participant_df)
    np.random.seed(42)
    for idx, (_, row) in enumerate(participant_df.iterrows()):
        angle = (idx / n_participants) * 2 * np.pi + np.random.uniform(-0.5, 0.5)
        radius = 60 + np.random.uniform(0, 120)
        # Use square aspect ratio for circular distribution
        x = 400 + radius * np.cos(angle)
        y = 190 + radius * np.sin(angle)
        dims = [float(row[c]) if c in row and pd.notna(row[c]) else 0.5 for c in dim_cols]
        uid = row['uid']
        matches = match_dict.get(uid, [])
        interactions = interaction_dict.get(uid, {'said_yes': [], 'received_yes': []})
        participants_data.append({
            'uid': uid, 'x': float(x), 'y': float(y), 'gender': int(row['gender']),
            'age': int(row['age']) if pd.notna(row.get('age')) else 0, 'dims': dims,
            'matches': matches, 'said_yes': interactions['said_yes'], 'received_yes': interactions['received_yes']
        })
    participants_json = json.dumps(participants_data)

    html_content = f'''
    <!DOCTYPE html><html><head><style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body, html {{ width: 100%; height: 100%; overflow: hidden; font-family: -apple-system, sans-serif; }}
    .nebula-container {{ width: 100%; height: 100%; min-height: 360px; background: white; position: relative; overflow: hidden; border-radius: 8px; border: 1px solid #eee; }}
    .nebula-canvas {{ width: 100%; height: 100%; display: block; }}
    .info-panel {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 8px; padding: 10px; color: #000; font-size: 11px; max-width: 200px; display: none; }}
    .info-panel h3 {{ color: #104a5b; margin-bottom: 8px; font-size: 12px; }}
    .info-panel .stat {{ margin: 5px 0; display: flex; align-items: center; }}
    .info-panel .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
    .legend {{ position: absolute; bottom: 10px; left: 10px; background: rgba(255,255,255,0.9); border: 1px solid #ccc; border-radius: 6px; padding: 8px; color: #000; font-size: 9px; }}
    .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
    .hint {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.8); border-radius: 6px; padding: 6px 10px; color: #666; font-size: 10px; }}
    .radar-tooltip {{ position: absolute; background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 6px; padding: 8px; color: #000; font-size: 10px; pointer-events: none; opacity: 0; z-index: 1000; }}
    </style></head><body>
    <div class="nebula-container">
        <canvas class="nebula-canvas" id="nebulaCanvas"></canvas>
        <div class="hint" id="hint">✨ Click star to explore</div>
        <div class="info-panel" id="infoPanel">
            <h3 id="selectedName">No Selection</h3>
            <div id="profileChars" style="margin:8px 0;padding:8px;background:#f9f9f9;border-radius:4px;"></div>
            <div class="stat"><span class="dot" style="background:#000;"></span><span>Matches: <strong id="matchCount">0</strong></span></div>
            <div class="stat"><span class="dot" style="background:#e63b55;"></span><span>You Yes: <strong id="yesCount">0</strong></span></div>
            <div class="stat"><span class="dot" style="background:#5f89d1;"></span><span>They Yes: <strong id="receivedCount">0</strong></span></div>
            <button onclick="clearSelection()" style="margin-top:8px;padding:4px 10px;background:#eee;border:none;color:#000;border-radius:4px;cursor:pointer;font-size:10px;">Clear</button>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#5f89d1;"></div>Male</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f2dada;"></div>Female</div>
            <div class="legend-item"><div class="legend-dot" style="background:#000;"></div>Match</div>
        </div>
        <div class="radar-tooltip" id="tooltip"></div>
    </div>
    <script>
        const participants = {participants_json};
        const canvas = document.getElementById('nebulaCanvas');
        const ctx = canvas.getContext('2d');
        const container = canvas.parentElement;
        const tooltip = document.getElementById('tooltip');
        const infoPanel = document.getElementById('infoPanel');
        const hint = document.getElementById('hint');
        let selectedParticipant = null;
        let stars = [];
        let animationFrame = 0;
        let positionMap = {{}};
        const baseSize = 380;
        participants.forEach((p) => {{
            p.phase = Math.random() * Math.PI * 2;
            p.phaseY = Math.random() * Math.PI * 2;
            p.speedX = 0.4 + Math.random() * 0.2;
            p.speedY = 0.3 + Math.random() * 0.2;
            p.moveRadius = 6 + Math.random() * 8;
        }});
        let time = 0;
        function resizeCanvas() {{
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            // Use uniform scale to maintain circular shape
            const scale = Math.min(canvas.width, canvas.height) / baseSize;
            const offsetX = (canvas.width - baseSize * scale) / 2;
            const offsetY = (canvas.height - baseSize * scale) / 2;
            participants.forEach(p => {{
                // Map from 800x380 base to actual canvas with uniform scaling
                const normalizedX = (p.x - 400) / 190;  // Normalize to -1 to 1
                const normalizedY = (p.y - 190) / 190;  // Normalize to -1 to 1
                p.baseX = canvas.width / 2 + normalizedX * (baseSize * scale / 2);
                p.baseY = canvas.height / 2 + normalizedY * (baseSize * scale / 2);
                if (p.currentX === undefined) {{ p.currentX = p.baseX; p.currentY = p.baseY; }}
            }});
        }}
        function updatePositions() {{
            time += 0.02;
            positionMap = {{}};
            participants.forEach(p => {{
                if (p.baseX === undefined) {{ p.baseX = p.x; p.baseY = p.y; }}
                p.currentX = p.baseX + Math.sin(time * p.speedX + p.phase) * p.moveRadius;
                p.currentY = p.baseY + Math.cos(time * p.speedY + p.phaseY) * p.moveRadius;
                positionMap[p.uid] = p;
            }});
        }}
        function draw() {{
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            drawNebulaClouds();
            drawBackgroundStars();
            if (selectedParticipant) drawConnections(selectedParticipant);
            participants.forEach(p => drawRadarGlyph(p));
        }}
        function generateStars() {{
            stars = [];
            for (let i = 0; i < 100; i++) {{
                stars.push({{ x: Math.random() * canvas.width, y: Math.random() * canvas.height, size: Math.random() * 1.2 + 0.3, twinkle: Math.random() * Math.PI * 2, speed: Math.random() * 0.02 + 0.01 }});
            }}
        }}
        function drawBackgroundStars() {{
            stars.forEach(star => {{
                const alpha = 0.2 + Math.sin(star.twinkle + animationFrame * star.speed) * 0.2;
                ctx.beginPath();
                ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(180,180,180,' + alpha + ')';
                ctx.fill();
            }});
        }}
        function drawNebulaClouds() {{
            const g1 = ctx.createRadialGradient(canvas.width*0.3, canvas.height*0.4, 0, canvas.width*0.3, canvas.height*0.4, 200);
            g1.addColorStop(0, 'rgba(95,137,209,0.04)'); g1.addColorStop(1, 'rgba(95,137,209,0)');
            ctx.fillStyle = g1; ctx.fillRect(0,0,canvas.width,canvas.height);
            const g2 = ctx.createRadialGradient(canvas.width*0.7, canvas.height*0.6, 0, canvas.width*0.7, canvas.height*0.6, 180);
            g2.addColorStop(0, 'rgba(242,218,218,0.04)'); g2.addColorStop(1, 'rgba(242,218,218,0)');
            ctx.fillStyle = g2; ctx.fillRect(0,0,canvas.width,canvas.height);
        }}
        function getParticipantColor(p) {{
            if (selectedParticipant) {{
                if (p.uid === selectedParticipant.uid) return '#104a5b';
                if (selectedParticipant.matches.includes(p.uid)) return '#b22222';
                if (selectedParticipant.said_yes.includes(p.uid)) return '#e63b55';
                if (selectedParticipant.received_yes.includes(p.uid)) return '#5f89d1';
                return '#a6c3d8';
            }}
            return p.gender === 1 ? '#5f89d1' : '#f2dada';
        }}
        function getParticipantOpacity(p) {{
            if (selectedParticipant) {{
                if (p.uid === selectedParticipant.uid) return 1;
                if (selectedParticipant.matches.includes(p.uid)) return 1;
                if (selectedParticipant.said_yes.includes(p.uid) || selectedParticipant.received_yes.includes(p.uid)) return 0.85;
                return 0.15;
            }}
            return 0.8;
        }}
        function drawRadarGlyph(p, size = 8) {{
            const color = getParticipantColor(p);
            const opacity = getParticipantOpacity(p);
            const dims = p.dims;
            const n = dims.length;
            const angleStep = (Math.PI * 2) / n;
            const px = p.currentX || p.baseX || p.x;
            const py = p.currentY || p.baseY || p.y;
            ctx.save();
            ctx.globalAlpha = opacity;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const angle = i * angleStep - Math.PI / 2;
                const r = (dims[i] * 0.7 + 0.3) * size;
                const x = px + r * Math.cos(angle);
                const y = py + r * Math.sin(angle);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 0.5;
            ctx.stroke();
            ctx.restore();
        }}
        function drawConnections(p) {{
            const px = p.currentX || p.baseX || p.x;
            const py = p.currentY || p.baseY || p.y;
            ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7; ctx.setLineDash([]);
            p.matches.forEach(uid => {{
                const t = positionMap[uid];
                if (t) {{ ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(t.currentX||t.baseX||t.x, t.currentY||t.baseY||t.y); ctx.stroke(); }}
            }});
            ctx.strokeStyle = '#e63b55'; ctx.lineWidth = 1; ctx.globalAlpha = 0.6; ctx.setLineDash([4,4]);
            p.said_yes.forEach(uid => {{
                if (!p.matches.includes(uid)) {{
                    const t = positionMap[uid];
                    if (t) {{ ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(t.currentX||t.baseX||t.x, t.currentY||t.baseY||t.y); ctx.stroke(); }}
                }}
            }});
            ctx.strokeStyle = '#5f89d1'; ctx.lineWidth = 1; ctx.globalAlpha = 0.6; ctx.setLineDash([2,3]);
            p.received_yes.forEach(uid => {{
                if (!p.matches.includes(uid) && !p.said_yes.includes(uid)) {{
                    const t = positionMap[uid];
                    if (t) {{ ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(t.currentX||t.baseX||t.x, t.currentY||t.baseY||t.y); ctx.stroke(); }}
                }}
            }});
            ctx.globalAlpha = 1.0; ctx.setLineDash([]);
        }}
        function getDistance(x, y, p) {{
            const px = p.currentX || p.baseX || p.x;
            const py = p.currentY || p.baseY || p.y;
            return Math.sqrt((x-px)*(x-px) + (y-py)*(y-py));
        }}
        resizeCanvas(); generateStars();
        function animate() {{ animationFrame++; updatePositions(); draw(); requestAnimationFrame(animate); }}
        animate();
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            let clicked = null;
            for (let i = participants.length - 1; i >= 0; i--) {{
                if (getDistance(x, y, participants[i]) < 12) {{ clicked = participants[i]; break; }}
            }}
            if (clicked) {{
                selectedParticipant = clicked;
                hint.style.display = 'none';
                infoPanel.style.display = 'block';
                document.getElementById('selectedName').textContent = clicked.uid + ' (' + (clicked.gender === 1 ? 'M' : 'F') + ', ' + (clicked.age || 'N/A') + ')';
                const dimLabels = ['Attr', 'Sinc', 'Intel', 'Fun', 'Amb'];
                let charsHtml = '';
                clicked.dims.forEach((d, i) => {{
                    const pct = Math.round(d * 100);
                    charsHtml += '<div style="display:flex;align-items:center;margin:2px 0;"><span style="width:40px;font-size:9px;">' + dimLabels[i] + '</span><div style="flex:1;height:5px;background:#eee;border-radius:2px;"><div style="width:' + pct + '%;height:100%;background:#5f89d1;border-radius:2px;"></div></div></div>';
                }});
                document.getElementById('profileChars').innerHTML = charsHtml;
                document.getElementById('matchCount').textContent = Math.ceil(clicked.matches.length / 2);
                document.getElementById('yesCount').textContent = clicked.said_yes.length;
                document.getElementById('receivedCount').textContent = clicked.received_yes.length;
            }} else {{
                selectedParticipant = null;
                hint.style.display = 'block';
                infoPanel.style.display = 'none';
            }}
        }});
        window.addEventListener('resize', () => {{ resizeCanvas(); generateStars(); }});
        function clearSelection() {{ selectedParticipant = null; hint.style.display = 'block'; infoPanel.style.display = 'none'; }}
    </script></body></html>
    '''
    return html_content

# ==================== Load and Process Data ====================
raw_df = load_data()
match_df = process_match_data(raw_df)
yes_edges, match_edges = build_network_data(match_df)

unique_interactions = match_df['pair_key_wave'].nunique()
df_matches = match_df[match_df['match'] == 1]
unique_match_pairs = df_matches['pair_key_wave'].nunique()
match_rate = (unique_match_pairs / unique_interactions * 100) if unique_interactions > 0 else 0

df_male = match_df[match_df['gender'] == 1]
df_female = match_df[match_df['gender'] == 0]
male_yes_rate = df_male['dec'].mean() * 100 if len(df_male) > 0 else 0
female_yes_rate = df_female['dec'].mean() * 100 if len(df_female) > 0 else 0

# Gender-specific decision breakdown (from male's perspective only to avoid double counting)
# Male says Yes, Female says No
male_yes_female_no = len(df_male[(df_male['dec'] == 1) & (df_male['dec_o'] == 0)])
# Female says Yes, Male says No
female_yes_male_no = len(df_male[(df_male['dec'] == 0) & (df_male['dec_o'] == 1)])
# Both say Yes (mutual match)
both_yes = len(df_male[(df_male['dec'] == 1) & (df_male['dec_o'] == 1)])
# Both say No
both_no = len(df_male[(df_male['dec'] == 0) & (df_male['dec_o'] == 0)])
total_raw = both_yes + male_yes_female_no + female_yes_male_no + both_no

persona_df = process_persona_data(raw_df)
persona_df = compute_persona_axes(persona_df)
persona_df['cluster'] = perform_clustering(persona_df, n_clusters=5)

# ==================== Dashboard Layout ====================
# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #104a5b 0%, #476d9e 100%); padding: 12px 20px; border-radius: 8px; margin-bottom: 10px;">
    <h1 style="color: white; margin: 0; font-size: 22px;">💫 Speed Dating Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Interactions", f"{unique_interactions:,}")
with m2:
    st.metric("Matches", f"{unique_match_pairs:,}")
with m3:
    st.metric("Match Rate", f"{match_rate:.1f}%")
with m4:
    st.metric("Male Yes", f"{male_yes_rate:.1f}%")
with m5:
    st.metric("Female Yes", f"{female_yes_rate:.1f}%")

# Main content - 3 columns
col1, col2, col3 = st.columns([1.2, 1.3, 1.5])

# Left: Sankey + Breakdown
with col1:
    st.markdown('<p class="section-title">Decision Flow</p>', unsafe_allow_html=True)
    fig_sankey = create_sankey(both_yes, male_yes_female_no, female_yes_male_no, both_no)
    st.plotly_chart(fig_sankey, use_container_width=True, config={'displayModeBar': False})

    # Breakdown bars with percentages
    st.markdown(f"""
    <div style="background:#fafafa;padding:10px;border-radius:6px;font-size:11px;">
        <div style="margin-bottom:8px;">
            <span style="color:#000;font-weight:bold;">{both_yes/total_raw*100:.1f}%</span> Mutual Match
            <div style="background:#ddd;border-radius:3px;height:6px;margin-top:3px;">
                <div style="background:#000;width:{both_yes/total_raw*100:.1f}%;height:100%;border-radius:3px;"></div>
            </div>
            <div style="color:#666;font-size:10px;margin-top:2px;">{both_yes//2:,} pairs</div>
        </div>
        <div style="margin-bottom:8px;">
            <span style="color:{MEDIUM_PINK};font-weight:bold;">{male_yes_female_no/total_raw*100:.1f}%</span> You Yes, They No
            <div style="background:#ddd;border-radius:3px;height:6px;margin-top:3px;">
                <div style="background:{MEDIUM_PINK};width:{male_yes_female_no/total_raw*100:.1f}%;height:100%;border-radius:3px;"></div>
            </div>
            <div style="color:#666;font-size:10px;margin-top:2px;">{male_yes_female_no:,} pairs</div>
        </div>
        <div style="margin-bottom:8px;">
            <span style="color:{MALE_COLOR};font-weight:bold;">{female_yes_male_no/total_raw*100:.1f}%</span> They Yes, You No
            <div style="background:#ddd;border-radius:3px;height:6px;margin-top:3px;">
                <div style="background:{MALE_COLOR};width:{female_yes_male_no/total_raw*100:.1f}%;height:100%;border-radius:3px;"></div>
            </div>
            <div style="color:#666;font-size:10px;margin-top:2px;">{female_yes_male_no:,} pairs</div>
        </div>
        <div>
            <span style="color:{GRID_COLOR};font-weight:bold;">{both_no/total_raw*100:.1f}%</span> Both No
            <div style="background:#ddd;border-radius:3px;height:6px;margin-top:3px;">
                <div style="background:{GRID_COLOR};width:{both_no/total_raw*100:.1f}%;height:100%;border-radius:3px;"></div>
            </div>
            <div style="color:#666;font-size:10px;margin-top:2px;">{both_no:,} pairs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Middle: Portrait Clustering
with col2:
    st.markdown('<p class="section-title">Portrait Clustering</p>', unsafe_allow_html=True)
    fig_persona = create_persona_scatter(persona_df)
    st.plotly_chart(fig_persona, use_container_width=True, config={'displayModeBar': False})

    # Compact cluster legend
    cluster_html = '<div style="display:flex;justify-content:space-around;font-size:9px;margin-top:-5px;">'
    for cid, name in CLUSTER_NAMES.items():
        count = (persona_df['cluster'] == cid).sum()
        pct = count / len(persona_df.dropna(subset=['cluster'])) * 100
        cluster_html += f'<div style="text-align:center;"><div style="width:10px;height:10px;background:{CLUSTER_COLORS[cid]};border-radius:50%;margin:0 auto;"></div><div>{name.split()[0][:6]}</div><div style="color:#888;">{pct:.0f}%</div></div>'
    cluster_html += '</div>'
    st.markdown(cluster_html, unsafe_allow_html=True)

# Right: Dating Network
with col3:
    st.markdown('<p class="section-title">Dating Network Mapping</p>', unsafe_allow_html=True)
    nebula_html = create_cosmic_nebula_html(match_df, match_edges, yes_edges)
    if nebula_html:
        components.html(nebula_html, height=380, scrolling=False)
