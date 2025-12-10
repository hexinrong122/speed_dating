import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ==================== Page Config ====================
st.set_page_config(
    page_title="Match Landscape Overview",
    page_icon="💘",
    layout="wide"
)

# ==================== Style Constants ====================
# Color palette as specified by user
BACKGROUND_COLOR = '#FFFFFF'  # White background
CARD_BG = '#FFFFFF'
MALE_COLOR = '#5f89d1'  # Gray blue
FEMALE_COLOR = '#f2dada'  # Light pink
MATCH_COLOR = '#000000'  # Black for mutual matches
GRID_COLOR = '#a6c3d8'   # Light gray blue
TEXT_COLOR = '#000000'   # Black text for better contrast on white

# Additional colors from palette
DARK_BLUE = '#104a5b'
MEDIUM_BLUE = '#476d9e'
LIGHT_GRAY_BLUE = '#a6c3d8'
LIGHT_TEAL = '#c5e9e3'
LIGHT_GRAY_PURPLE = '#d7e1ab'
MEDIUM_PINK = '#e63b55'
DARK_PINK = '#e89090'
DARK_RED = '#b22222'

# ==================== Data Loading ====================
@st.cache_data
def load_and_clean_data():
    """Load and clean speed dating data with enhanced processing"""
    df = pd.read_csv("data/Speed Dating Data.csv", encoding='latin-1')

    # Select relevant columns - add self-perception for radar charts
    cols = ['iid', 'pid', 'gender', 'wave', 'match', 'dec', 'dec_o', 'age', 'age_o',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()

    # Remove rows with missing critical values
    df = df.dropna(subset=['dec', 'dec_o', 'iid', 'pid'])

    # Convert types
    df['iid'] = df['iid'].astype(int)
    df['pid'] = df['pid'].astype(int)
    df['dec'] = df['dec'].astype(int)
    df['dec_o'] = df['dec_o'].astype(int)
    df['wave'] = df['wave'].astype(int)
    df['gender'] = df['gender'].astype(int)

    # Rebuild match
    df['match'] = ((df['dec'] == 1) & (df['dec_o'] == 1)).astype(int)

    # Create unique IDs with wave prefix for network
    df['uid'] = df['wave'].astype(str) + '_' + df['iid'].astype(str)
    df['uid_partner'] = df['wave'].astype(str) + '_' + df['pid'].astype(str)

    # Create pair key for deduplication (unordered pair identifier)
    df['pair_key'] = df.apply(lambda r: tuple(sorted([r['iid'], r['pid']])), axis=1)
    df['pair_key_wave'] = df['wave'].astype(str) + '_' + df['pair_key'].astype(str)

    return df

@st.cache_data
def build_network_data(df):
    """Build network edges and node statistics"""
    # Directed edges (A says Yes to B)
    yes_edges = df[df['dec'] == 1][['uid', 'uid_partner', 'gender', 'wave']].copy()
    yes_edges.columns = ['source', 'target', 'source_gender', 'wave']

    # Mutual match edges
    match_edges = df[df['match'] == 1][['uid', 'uid_partner', 'wave']].copy()
    match_edges.columns = ['source', 'target', 'wave']

    # Node statistics
    # In-degree: how many people said Yes to this person
    in_degree = df.groupby('uid_partner')['dec'].sum().reset_index()
    in_degree.columns = ['uid', 'in_degree']

    # Out-degree: how many people this person said Yes to
    out_degree = df.groupby('uid')['dec'].sum().reset_index()
    out_degree.columns = ['uid', 'out_degree']

    # Node info
    node_info = df.groupby('uid').first()[['gender', 'wave', 'age']].reset_index()

    # Merge
    nodes = node_info.merge(in_degree, on='uid', how='left')
    nodes = nodes.merge(out_degree, on='uid', how='left')
    nodes = nodes.fillna(0)

    return yes_edges, match_edges, nodes

# ==================== Visualization Functions ====================
def create_radial_gauge(match_rate, male_interactions, female_interactions, stats):
    """Create radial gauge with outer ring for interactions and inner ring for match rate"""
    fig = go.Figure()

    total_interactions = male_interactions + female_interactions
    male_pct = (male_interactions / total_interactions * 100) if total_interactions > 0 else 50
    female_pct = (female_interactions / total_interactions * 100) if total_interactions > 0 else 50

    # Outer ring - Total interactions (Male blue + Female pink)
    fig.add_trace(go.Pie(
        values=[male_interactions, female_interactions],
        hole=0.70,
        marker=dict(
            colors=[MALE_COLOR, FEMALE_COLOR],
            line=dict(color=BACKGROUND_COLOR, width=2)
        ),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Interactions: %{value:,}<br>Ratio: %{percent}<extra></extra>',
        labels=['Male', 'Female'],
        showlegend=False,
        domain=dict(x=[0.10, 0.90], y=[0.10, 0.90])
    ))

    # Inner ring - Match success rate
    fig.add_trace(go.Pie(
        values=[match_rate, 100 - match_rate],
        hole=0.80,
        marker=dict(
            colors=[MATCH_COLOR, CARD_BG],
            line=dict(color=BACKGROUND_COLOR, width=1)
        ),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
        labels=['Match Rate', 'No Match'],
        showlegend=False,
        domain=dict(x=[0.22, 0.78], y=[0.22, 0.78])
    ))

    # Center text - Match Rate
    fig.add_annotation(
        x=0.5, y=0.55,
        text=f"<b>{match_rate:.1f}%</b>",
        font=dict(size=32, color=MATCH_COLOR, family="Arial Black"),
        showarrow=False
    )
    fig.add_annotation(
        x=0.5, y=0.42,
        text="Match Rate",
        font=dict(size=11, color=TEXT_COLOR),
        showarrow=False
    )

    # Legend annotations
    fig.add_annotation(
        x=0.12, y=0.95,
        text=f"<span style='color:{MALE_COLOR}'>●</span> Male ({male_pct:.0f}%)",
        font=dict(size=10, color=TEXT_COLOR),
        showarrow=False,
        xanchor='left'
    )
    fig.add_annotation(
        x=0.12, y=0.88,
        text=f"<span style='color:{FEMALE_COLOR}'>●</span> Female ({female_pct:.0f}%)",
        font=dict(size=10, color=TEXT_COLOR),
        showarrow=False,
        xanchor='left'
    )
    fig.add_annotation(
        x=0.12, y=0.81,
        text=f"<span style='color:{MATCH_COLOR}'>●</span> Match Success",
        font=dict(size=10, color=TEXT_COLOR),
        showarrow=False,
        xanchor='left'
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
    )

    return fig

def prepare_participant_profiles(df):
    """Prepare participant profiles with self-perception dimensions for radar stars"""
    # Self-perception columns
    self_cols = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
    available_cols = [c for c in self_cols if c in df.columns]

    if not available_cols:
        return None

    # Get participant-level data
    participant_df = df.groupby('uid').first().reset_index()

    # Normalize dimensions to 0-1 for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Fill NaN with median
    for col in available_cols:
        if col in participant_df.columns:
            participant_df[col] = pd.to_numeric(participant_df[col], errors='coerce')
            participant_df[col] = participant_df[col].fillna(participant_df[col].median())

    if len(participant_df) > 0 and all(c in participant_df.columns for c in available_cols):
        participant_df[available_cols] = scaler.fit_transform(participant_df[available_cols])

    return participant_df, available_cols

def create_cosmic_nebula_html(df, match_edges, yes_edges):
    """
    Create an interactive cosmic nebula visualization using HTML/JavaScript.
    All participants are shown as radar-shaped stars with click interaction.
    """
    import json

    # Get participant profiles
    result = prepare_participant_profiles(df)
    if result is None:
        return None

    participant_df, dim_cols = result

    # Build interaction data
    match_dict = {}
    for _, row in match_edges.iterrows():
        src, tgt = row['source'], row['target']
        if src not in match_dict:
            match_dict[src] = []
        if tgt not in match_dict:
            match_dict[tgt] = []
        match_dict[src].append(tgt)
        match_dict[tgt].append(src)

    interaction_dict = {}
    for _, row in yes_edges.iterrows():
        src, tgt = row['source'], row['target']
        if src not in interaction_dict:
            interaction_dict[src] = {'said_yes': [], 'received_yes': []}
        interaction_dict[src]['said_yes'].append(tgt)

    for _, row in df[df['dec_o'] == 1].iterrows():
        uid = row['uid']
        partner = row['uid_partner']
        if uid not in interaction_dict:
            interaction_dict[uid] = {'said_yes': [], 'received_yes': []}
        interaction_dict[uid]['received_yes'].append(partner)

    # Prepare participant data for JavaScript
    participants_data = []
    n_participants = len(participant_df)

    # Generate positions with wider distribution
    np.random.seed(42)
    
    for idx, (_, row) in enumerate(participant_df.iterrows()):
        # Distribute stars in a wider circular pattern
        angle = (idx / n_participants) * 2 * np.pi + np.random.uniform(-0.5, 0.5)
        # Varying radius for natural look, spread between 100 and 300 pixels from center
        radius = 100 + np.random.uniform(0, 200)

        # Center the distribution in the canvas (assuming 900x500 canvas)
        x = 450 + radius * np.cos(angle)
        y = 250 + radius * np.sin(angle)

        # Get dimension values
        dims = [float(row[c]) if c in row and pd.notna(row[c]) else 0.5 for c in dim_cols]

        uid = row['uid']
        matches = match_dict.get(uid, [])
        interactions = interaction_dict.get(uid, {'said_yes': [], 'received_yes': []})

        participants_data.append({
            'uid': uid,
            'x': float(x),
            'y': float(y),
            'gender': int(row['gender']),
            'age': int(row['age']) if pd.notna(row.get('age')) else 0,
            'dims': dims,
            'matches': matches,
            'said_yes': interactions['said_yes'],
            'received_yes': interactions['received_yes']
        })

    # Convert to JSON
    participants_json = json.dumps(participants_data)

    # Generate HTML/JavaScript
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body, html {{
                width: 100%;
                height: 100%;
                overflow: hidden;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            .nebula-container {{
                width: 100%;
                height: 100%;
                min-height: 500px;
                background: white;
                position: relative;
                overflow: hidden;
                border-radius: 12px;
                border: 1px solid #eeeeee;
            }}
            .nebula-canvas {{
                width: 100%;
                height: 100%;
                display: block;
            }}
            .info-panel {{
                position: absolute;
                top: 15px;
                right: 15px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #cccccc;
                border-radius: 10px;
                padding: 15px;
                color: #000000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 12px;
                max-width: 250px;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }}
            .info-panel h3 {{
                color: #104a5b;
                margin-bottom: 10px;
                font-size: 14px;
            }}
            .info-panel .stat {{
                margin: 8px 0;
                display: flex;
                align-items: center;
            }}
            .info-panel .dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
                display: inline-block;
            }}
            .legend {{
                position: absolute;
                bottom: 15px;
                left: 15px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 12px;
                color: #000000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 11px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 5px 0;
            }}
            .legend-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .hint {{
                position: absolute;
                top: 15px;
                left: 15px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 8px;
                padding: 10px 15px;
                color: #666666;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 11px;
            }}
            .radar-tooltip {{
                position: absolute;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 10px;
                color: #000000;
                font-size: 11px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
                z-index: 1000;
            }}
        </style>
    </head>
    <body>
        <div class="nebula-container">
            <canvas class="nebula-canvas" id="nebulaCanvas"></canvas>
            <div class="hint" id="hint">✨ Click any star to explore connections</div>
            <div class="info-panel" id="infoPanel" style="display: none;">
                <h3 id="selectedName">No Selection</h3>
                <div id="profileChars" style="margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 5px;"></div>
                <div class="stat">
                    <span class="dot" style="background: #000000;"></span>
                    <span>Mutual Matches: <strong id="matchCount">0</strong></span>
                </div>
                <div class="stat">
                    <span class="dot" style="background: #e63b55;"></span>
                    <span>You Said Yes: <strong id="yesCount">0</strong></span>
                </div>
                <div class="stat">
                    <span class="dot" style="background: #5f89d1;"></span>
                    <span>They Said Yes: <strong id="receivedCount">0</strong></span>
                </div>
                <button onclick="clearSelection()" style="margin-top: 10px; padding: 5px 15px; background: #eeeeee; border: none; color: #000000; border-radius: 5px; cursor: pointer;">Clear Selection</button>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background: #5f89d1;"></div>Male</div>
                <div class="legend-item"><div class="legend-dot" style="background: #f2dada;"></div>Female</div>
                <div class="legend-item"><div class="legend-dot" style="background: #000000;"></div>Mutual Match</div>
                <div class="legend-item"><div class="legend-dot" style="background: #e63b55;"></div>You Said Yes</div>
                <div class="legend-item"><div class="legend-dot" style="background: #5f89d1;"></div>They Said Yes</div>
                <div class="legend-item"><div class="legend-dot" style="background: #a6c3d8;"></div>No Interaction</div>
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
            let hoveredParticipant = null;
            let stars = [];
            let animationFrame = 0;
            let positionMap = {{}};

            // Store original positions for subtle movement - scale to canvas center
            const baseWidth = 900;
            const baseHeight = 500;

            // Add floating animation parameters for each participant
            participants.forEach((p, idx) => {{
                p.phase = Math.random() * Math.PI * 2;
                p.phaseY = Math.random() * Math.PI * 2;
                p.speedX = 0.5 + Math.random() * 0.3;  // Faster speed
                p.speedY = 0.4 + Math.random() * 0.3;  // Faster speed
                // Add individual movement patterns
                p.moveRadius = 8 + Math.random() * 12; // Movement radius between 8-20 pixels
            }});

            // Floating animation
            let time = 0;

            function resizeCanvas() {{
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;

                // Recalculate positions based on canvas size
                const scaleX = canvas.width / baseWidth;
                const scaleY = canvas.height / baseHeight;

                participants.forEach(p => {{
                    p.baseX = p.x * scaleX;
                    p.baseY = p.y * scaleY;
                    // Initialize current positions
                    if (p.currentX === undefined) {{
                        p.currentX = p.baseX;
                        p.currentY = p.baseY;
                    }}
                }});
            }}

            function updatePositions() {{
                time += 0.03; // Animation speed
                positionMap = {{}};

                participants.forEach(p => {{
                    // Store base positions if not already stored
                    if (p.baseX === undefined || p.baseY === undefined) {{
                        p.baseX = p.x;
                        p.baseY = p.y;
                    }}

                    // Calculate floating positions using sine/cosine for smooth natural movement
                    p.currentX = p.baseX + Math.sin(time * p.speedX + p.phase) * p.moveRadius;
                    p.currentY = p.baseY + Math.cos(time * p.speedY + p.phaseY) * p.moveRadius;

                    positionMap[p.uid] = p;
                }});
            }}

            function draw() {{
                // Clear canvas with white background
                ctx.fillStyle = '#FFFFFF';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw subtle background effects
                drawNebulaClouds();
                drawBackgroundStars();

                // Draw connection lines first (so they appear behind nodes)
                if (selectedParticipant) {{
                    drawConnections(selectedParticipant);
                }}

                // Draw all participants as radar glyphs
                participants.forEach(p => {{
                    drawRadarGlyph(p);
                }});
            }}

            // Generate background stars
            function generateStars() {{
                stars = [];
                for (let i = 0; i < 150; i++) {{
                    stars.push({{
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        size: Math.random() * 1.5 + 0.5,
                        twinkle: Math.random() * Math.PI * 2,
                        speed: Math.random() * 0.02 + 0.01
                    }});
                }}
            }}

            function drawBackgroundStars() {{
                stars.forEach(star => {{
                    const alpha = 0.3 + Math.sin(star.twinkle + animationFrame * star.speed) * 0.3;
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(180, 180, 180, ' + alpha + ')';
                    ctx.fill();
                }});
            }}

            function drawNebulaClouds() {{
                // Draw subtle nebula effect
                const gradient1 = ctx.createRadialGradient(
                    canvas.width * 0.3, canvas.height * 0.4, 0,
                    canvas.width * 0.3, canvas.height * 0.4, 300
                );
                gradient1.addColorStop(0, 'rgba(95, 137, 209, 0.05)');
                gradient1.addColorStop(1, 'rgba(95, 137, 209, 0)');
                ctx.fillStyle = gradient1;
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                const gradient2 = ctx.createRadialGradient(
                    canvas.width * 0.7, canvas.height * 0.6, 0,
                    canvas.width * 0.7, canvas.height * 0.6, 250
                );
                gradient2.addColorStop(0, 'rgba(242, 218, 218, 0.05)');
                gradient2.addColorStop(1, 'rgba(242, 218, 218, 0)');
                ctx.fillStyle = gradient2;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
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
                    if (selectedParticipant.said_yes.includes(p.uid) ||
                        selectedParticipant.received_yes.includes(p.uid)) return 0.9;
                    return 0.2;
                }}
                return 0.85;
            }}

            function drawRadarGlyph(p, size = 10) {{
                const color = getParticipantColor(p);
                const opacity = getParticipantOpacity(p);
                const dims = p.dims;
                const n = dims.length;
                const angleStep = (Math.PI * 2) / n;
                // Use animated positions
                const px = p.currentX || p.baseX || p.x;
                const py = p.currentY || p.baseY || p.y;

                ctx.save();
                ctx.globalAlpha = opacity;

                // Draw radar polygon
                ctx.beginPath();
                for (let i = 0; i < n; i++) {{
                    const angle = i * angleStep - Math.PI / 2;
                    const r = (dims[i] * 0.7 + 0.3) * size;
                    const x = px + r * Math.cos(angle);
                    const y = py + r * Math.sin(angle);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.restore();
            }}

            function drawConnections(p) {{
                const px = p.currentX || p.baseX || p.x;
                const py = p.currentY || p.baseY || p.y;

                // Draw mutual matches (black)
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.8;
                ctx.setLineDash([]);
                p.matches.forEach(uid => {{
                    const target = positionMap[uid];
                    if (target) {{
                        const tx = target.currentX || target.baseX || target.x;
                        const ty = target.currentY || target.baseY || target.y;
                        ctx.beginPath();
                        ctx.moveTo(px, py);
                        ctx.lineTo(tx, ty);
                        ctx.stroke();
                    }}
                }});

                // Draw "You said Yes" connections (red)
                ctx.strokeStyle = '#e63b55';
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = 0.7;
                ctx.setLineDash([5, 5]);
                p.said_yes.forEach(uid => {{
                    if (!p.matches.includes(uid)) {{
                        const target = positionMap[uid];
                        if (target) {{
                            const tx = target.currentX || target.baseX || target.x;
                            const ty = target.currentY || target.baseY || target.y;
                            ctx.beginPath();
                            ctx.moveTo(px, py);
                            ctx.lineTo(tx, ty);
                            ctx.stroke();
                        }}
                    }}
                }});

                // Draw "They said Yes" connections (blue)
                ctx.strokeStyle = '#5f89d1';
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = 0.7;
                ctx.setLineDash([2, 4]);
                p.received_yes.forEach(uid => {{
                    if (!p.matches.includes(uid) && !p.said_yes.includes(uid)) {{
                        const target = positionMap[uid];
                        if (target) {{
                            const tx = target.currentX || target.baseX || target.x;
                            const ty = target.currentY || target.baseY || target.y;
                            ctx.beginPath();
                            ctx.moveTo(px, py);
                            ctx.lineTo(tx, ty);
                            ctx.stroke();
                        }}
                    }}
                }});
                ctx.globalAlpha = 1.0;
                ctx.setLineDash([]);
            }}

            function getDistance(x, y, p) {{
                const px = p.currentX || p.baseX || p.x;
                const py = p.currentY || p.baseY || p.y;
                const dx = x - px;
                const dy = y - py;
                return Math.sqrt(dx * dx + dy * dy);
            }}

            // Initial setup
            resizeCanvas();
            generateStars();

            // Animation loop
            function animate() {{
                animationFrame++;
                updatePositions();
                draw();
                requestAnimationFrame(animate);
            }}

            // Start animation
            animate();

            // Event listeners
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                // Find hovered participant
                hoveredParticipant = null;
                for (let i = participants.length - 1; i >= 0; i--) {{
                    const p = participants[i];
                    if (getDistance(x, y, p) < 15) {{
                        hoveredParticipant = p;
                        break;
                    }}
                }}

                // Update tooltip with characteristics
                if (hoveredParticipant) {{
                    const px = hoveredParticipant.currentX || hoveredParticipant.baseX || hoveredParticipant.x;
                    const py = hoveredParticipant.currentY || hoveredParticipant.baseY || hoveredParticipant.y;
                    const dimLabels = ['Attractive', 'Sincere', 'Intelligent', 'Fun', 'Ambitious'];
                    let dimsHtml = '';
                    hoveredParticipant.dims.forEach((d, i) => {{
                        const pct = Math.round(d * 100);
                        dimsHtml += '<div style="display:flex;align-items:center;margin:2px 0;">' +
                            '<span style="width:70px;font-size:10px;">' + dimLabels[i] + ':</span>' +
                            '<div style="flex:1;height:6px;background:#eee;border-radius:3px;margin-left:5px;">' +
                            '<div style="width:' + pct + '%;height:100%;background:#5f89d1;border-radius:3px;"></div>' +
                            '</div>' +
                            '<span style="width:30px;text-align:right;font-size:10px;">' + pct + '%</span>' +
                            '</div>';
                    }});
                    tooltip.style.opacity = '1';
                    tooltip.style.left = (px + 20) + 'px';
                    tooltip.style.top = (py - 60) + 'px';
                    tooltip.innerHTML = '<div style="font-weight:bold;margin-bottom:5px;">' + hoveredParticipant.uid + '</div>' +
                        '<div style="font-size:11px;color:#666;margin-bottom:8px;">' +
                        (hoveredParticipant.gender === 1 ? 'Male' : 'Female') + ', Age: ' + (hoveredParticipant.age || 'N/A') +
                        '</div>' +
                        '<div style="border-top:1px solid #eee;padding-top:5px;">' + dimsHtml + '</div>';
                }} else {{
                    tooltip.style.opacity = '0';
                }}
            }});

            canvas.addEventListener('click', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                // Find clicked participant
                let clickedParticipant = null;
                for (let i = participants.length - 1; i >= 0; i--) {{
                    const p = participants[i];
                    if (getDistance(x, y, p) < 15) {{
                        clickedParticipant = p;
                        break;
                    }}
                }}

                if (clickedParticipant) {{
                    selectedParticipant = clickedParticipant;
                    hint.style.display = 'none';
                    infoPanel.style.display = 'block';

                    // Update info panel
                    document.getElementById('selectedName').textContent =
                        selectedParticipant.uid + ' (' +
                        (selectedParticipant.gender === 1 ? 'Male' : 'Female') +
                        ', Age: ' + (selectedParticipant.age || 'N/A') + ')';

                    // Build characteristics display
                    const dimLabels = ['Attractive', 'Sincere', 'Intelligent', 'Fun', 'Ambitious'];
                    const colors = ['#e63b55', '#5f89d1', '#104a5b', '#f2dada', '#476d9e'];
                    let charsHtml = '<div style="font-size:11px;color:#666;margin-bottom:8px;">Self-Perception:</div>';
                    selectedParticipant.dims.forEach((d, i) => {{
                        const pct = Math.round(d * 100);
                        charsHtml += '<div style="display:flex;align-items:center;margin:4px 0;">' +
                            '<span style="width:75px;font-size:11px;">' + dimLabels[i] + '</span>' +
                            '<div style="flex:1;height:8px;background:#eee;border-radius:4px;margin:0 8px;">' +
                            '<div style="width:' + pct + '%;height:100%;background:' + colors[i] + ';border-radius:4px;"></div>' +
                            '</div>' +
                            '<span style="width:35px;text-align:right;font-size:11px;font-weight:bold;">' + pct + '%</span>' +
                            '</div>';
                    }});
                    document.getElementById('profileChars').innerHTML = charsHtml;

                    document.getElementById('matchCount').textContent = Math.ceil(selectedParticipant.matches.length / 2);
                    document.getElementById('yesCount').textContent = selectedParticipant.said_yes.length;
                    document.getElementById('receivedCount').textContent = selectedParticipant.received_yes.length;
                }} else {{
                    selectedParticipant = null;
                    hint.style.display = 'block';
                    infoPanel.style.display = 'none';
                }}
            }});

            window.addEventListener('resize', () => {{
                resizeCanvas();
                generateStars();
            }});

            function clearSelection() {{
                selectedParticipant = null;
                hint.style.display = 'block';
                infoPanel.style.display = 'none';
            }}
        </script>
    </body>
    </html>
    '''

    return html_content

def create_decision_breakdown_sankey(both_yes, you_yes_they_no, you_no_they_yes, both_no):
    """Create Sankey-style flow diagram for decision breakdown"""
    # Fix the logic: both_yes should be divided by 2 since each match creates two records
    both_yes_corrected = both_yes // 2

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        textfont=dict(color='#333333', size=11, family='Arial'),
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color='#666666', width=1),
            label=['All Interactions', 'You Say Yes', 'You Say No',
                   'Mutual Match', 'You Yes, They No', 'They Yes, You No', 'Both No'],
            # 浅粉色, 浅粉色, 浅灰色, 深粉色, 浅粉色, 中灰色, 深灰色
            color=['#f2dada', '#f2dada', '#cccccc', '#e63b55', '#f2dada', '#999999', '#666666'],
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 5, 6],
            value=[both_yes_corrected + you_yes_they_no,
                   you_no_they_yes + both_no,
                   both_yes_corrected,
                   you_yes_they_no,
                   you_no_they_yes,
                   both_no],
            color=[
                'rgba(242, 218, 218, 0.6)',  # 浅粉色 -> You Say Yes
                'rgba(204, 204, 204, 0.5)',  # 浅灰色 -> You Say No
                'rgba(230, 59, 85, 0.6)',    # 深粉色 -> Mutual Match
                'rgba(242, 218, 218, 0.5)',  # 浅粉色 -> You Yes, They No
                'rgba(153, 153, 153, 0.5)',  # 中灰色 -> They Yes, You No
                'rgba(102, 102, 102, 0.5)'   # 深灰色 -> Both No
            ]
        )
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text="Decision Flow Analysis",
            font=dict(color='#333333', size=14, family='Arial'),
            x=0.5
        )
    )

    return fig

# ==================== Load Data ====================
df = load_and_clean_data()
yes_edges, match_edges, nodes = build_network_data(df)

# ==================== Sidebar ====================
st.sidebar.header("🎛️ Controls")

# Wave filter
waves = ['All'] + sorted(df['wave'].unique().tolist())
selected_wave = st.sidebar.selectbox("Filter by Wave", options=waves, index=0)

# ==================== Filter Data ====================
if selected_wave != 'All':
    df_filtered = df[df['wave'] == selected_wave].copy()
else:
    df_filtered = df.copy()

# ==================== Calculate Metrics ====================
# Deduplicated total interactions (unique pairs)
unique_interactions = df_filtered['pair_key_wave'].nunique()

# Deduplicated mutual matches (unique matching pairs)
df_matches = df_filtered[df_filtered['match'] == 1]
unique_match_pairs = df_matches['pair_key_wave'].nunique()

# Match rate based on deduplicated data
match_rate = (unique_match_pairs / unique_interactions * 100) if unique_interactions > 0 else 0

# Gender-based interaction counts (for radial gauge outer ring)
male_interactions = len(df_filtered[df_filtered['gender'] == 1])
female_interactions = len(df_filtered[df_filtered['gender'] == 0])

# Gender yes rates (still based on raw data for accuracy)
df_male = df_filtered[df_filtered['gender'] == 1]
df_female = df_filtered[df_filtered['gender'] == 0]
male_yes_rate = df_male['dec'].mean() * 100 if len(df_male) > 0 else 0
female_yes_rate = df_female['dec'].mean() * 100 if len(df_female) > 0 else 0

# Decision breakdown (based on raw data)
both_yes = len(df_filtered[(df_filtered['dec'] == 1) & (df_filtered['dec_o'] == 1)])
you_yes_they_no = len(df_filtered[(df_filtered['dec'] == 1) & (df_filtered['dec_o'] == 0)])
you_no_they_yes = len(df_filtered[(df_filtered['dec'] == 0) & (df_filtered['dec_o'] == 1)])
both_no = len(df_filtered[(df_filtered['dec'] == 0) & (df_filtered['dec_o'] == 0)])

# ==================== Header ====================
st.markdown(f"""
<div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #eeeeee;">
    <h1 style="color: black; margin: 0;">💘 Match Landscape Overview</h1>
    <p style="color: {TEXT_COLOR}; margin: 5px 0 0 0;">Understanding the macro landscape of speed dating: match is rare, gender preferences differ significantly</p>
</div>
""", unsafe_allow_html=True)

# ==================== Top Row: Metrics ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="background: {CARD_BG}; padding: 15px; border-radius: 8px; border-left: 4px solid {MATCH_COLOR}; border: 1px solid #eeeeee;">
        <p style="color: {TEXT_COLOR}; margin: 0; font-size: 12px;">TOTAL INTERACTIONS</p>
        <h2 style="color: black; margin: 5px 0;">{unique_interactions:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: {CARD_BG}; padding: 15px; border-radius: 8px; border-left: 4px solid {DARK_RED}; border: 1px solid #eeeeee;">
        <p style="color: {TEXT_COLOR}; margin: 0; font-size: 12px;">MUTUAL MATCHES</p>
        <h2 style="color: black; margin: 5px 0;">{unique_match_pairs:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background: {CARD_BG}; padding: 15px; border-radius: 8px; border-left: 4px solid {MALE_COLOR}; border: 1px solid #eeeeee;">
        <p style="color: {TEXT_COLOR}; margin: 0; font-size: 12px;">MALE → FEMALE YES</p>
        <h2 style="color: {MALE_COLOR}; margin: 5px 0;">{male_yes_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="background: {CARD_BG}; padding: 15px; border-radius: 8px; border-left: 4px solid {FEMALE_COLOR}; border: 1px solid #eeeeee;">
        <p style="color: {TEXT_COLOR}; margin: 0; font-size: 12px;">FEMALE → MALE YES</p>
        <h2 style="color: {FEMALE_COLOR}; margin: 5px 0;">{female_yes_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== Decision Flow ====================
st.markdown(f"<h4 style='color: {TEXT_COLOR};'>Decision Outcome Distribution</h4>", unsafe_allow_html=True)

# Total raw interactions for breakdown percentages
total_raw = both_yes + you_yes_they_no + you_no_they_yes + both_no

# Sankey diagram and breakdown side by side
col_sankey, col_breakdown = st.columns([2, 1])

with col_sankey:
    fig_sankey = create_decision_breakdown_sankey(both_yes, you_yes_they_no, you_no_they_yes, both_no)
    st.plotly_chart(fig_sankey, use_container_width=True, theme=None)

with col_breakdown:
    # Breakdown metrics
    st.markdown(f"""
    <div style="background: {CARD_BG}; padding: 20px; border-radius: 8px; border: 1px solid #eeeeee;">
        <div style="margin-bottom: 15px;">
            <span style="color: #000000; font-size: 24px; font-weight: bold;">{both_yes//2:,}</span>
            <span style="color: {TEXT_COLOR};"> Mutual Matches</span>
            <div style="background: {GRID_COLOR}; border-radius: 4px; height: 8px; margin-top: 5px;">
                <div style="background: #000000; width: {both_yes/total_raw*100:.1f}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="color: {TEXT_COLOR}; font-size: 11px;">{both_yes/total_raw*100:.1f}%</span>
        </div>
        <div style="margin-bottom: 15px;">
            <span style="color: {MEDIUM_PINK}; font-size: 24px; font-weight: bold;">{you_yes_they_no:,}</span>
            <span style="color: {TEXT_COLOR};"> You Yes, They No</span>
            <div style="background: {GRID_COLOR}; border-radius: 4px; height: 8px; margin-top: 5px;">
                <div style="background: {MEDIUM_PINK}; width: {you_yes_they_no/total_raw*100:.1f}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="color: {TEXT_COLOR}; font-size: 11px;">{you_yes_they_no/total_raw*100:.1f}%</span>
        </div>
        <div style="margin-bottom: 15px;">
            <span style="color: {MALE_COLOR}; font-size: 24px; font-weight: bold;">{you_no_they_yes:,}</span>
            <span style="color: {TEXT_COLOR};"> They Yes, You No</span>
            <div style="background: {GRID_COLOR}; border-radius: 4px; height: 8px; margin-top: 5px;">
                <div style="background: {MALE_COLOR}; width: {you_no_they_yes/total_raw*100:.1f}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="color: {TEXT_COLOR}; font-size: 11px;">{you_no_they_yes/total_raw*100:.1f}%</span>
        </div>
        <div>
            <span style="color: {GRID_COLOR}; font-size: 24px; font-weight: bold;">{both_no:,}</span>
            <span style="color: {TEXT_COLOR};"> Both No</span>
            <div style="background: {GRID_COLOR}; border-radius: 4px; height: 8px; margin-top: 5px;">
                <div style="background: {TEXT_COLOR}; width: {both_no/total_raw*100:.1f}%; height: 100%; border-radius: 4px; opacity: 0.3;"></div>
            </div>
            <span style="color: {TEXT_COLOR}; font-size: 11px;">{both_no/total_raw*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== Main Content ====================
# Cosmic Nebula (now below the sankey diagram)
st.markdown(f"<h4 style='color: {TEXT_COLOR};'>✨ Cosmic Personality Nebula</h4>", unsafe_allow_html=True)

# Filter data by wave for nebula
wave_for_network = None if selected_wave == 'All' else selected_wave

if wave_for_network:
    df_nebula = df_filtered.copy()
    match_edges_nebula = match_edges[match_edges['wave'] == wave_for_network]
    yes_edges_nebula = yes_edges[yes_edges['wave'] == wave_for_network]
else:
    df_nebula = df.copy()
    match_edges_nebula = match_edges.copy()
    yes_edges_nebula = yes_edges.copy()

# Create cosmic nebula HTML component
import streamlit.components.v1 as components

nebula_html = create_cosmic_nebula_html(df_nebula, match_edges_nebula, yes_edges_nebula)

if nebula_html:
    components.html(nebula_html, height=520, scrolling=False)
else:
    st.warning("Unable to create nebula visualization. Check if personality data is available.")

# ==================== Footer ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: {TEXT_COLOR}; font-size: 11px; padding: 20px;">
    <b>Data Source:</b> Speed Dating Experiment Dataset |
    <b>Note:</b> Each interaction generates two records (one per participant)
</div>
""", unsafe_allow_html=True)