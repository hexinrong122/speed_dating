# Speed Dating Analysis Dashboard

This repository contains a Streamlit dashboard for analyzing the Speed Dating Experiment dataset. The dashboard provides insights into dating preferences, match outcomes, and participant characteristics through interactive visualizations.

## Features

- **Match Landscape Overview**: Analyze match rates, gender-based decision patterns, and overall statistics from the speed dating experiment
- **Interactive Visualizations**: Explore data through various charts and graphs
- **Data Insights**: Gain understanding of factors affecting match decisions in speed dating events

## Prerequisites

- Python 3.7 or higher
- Pip package manager

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/hexinrong122/speed_dating.git
   cd speed_dating
   ```
   or open this folder

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit dashboard, execute:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser. 

To view the static HTML visualizations:
- Open `Triangle Interaction Plot.html` directly in your web browser
- Open `Interaction Trajectory Map Attraction Over Time.html` directly in your web browser

To run the Jupyter notebook:
- Open `line chart+heat map.ipynb` in Jupyter Notebook or Jupyter Lab
- Execute the cells to generate line charts and heat maps for attribute ratings over time

## Dataset

The dashboard uses the Speed Dating Experiment dataset from Columbia Business School. The dataset includes information about participants' demographics, preferences, and match outcomes from speed dating events.

## Project Structure

```
speed_dating/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation
├── Triangle Interaction Plot.html     # Static visualization of relationship triangle
├── Interaction Trajectory Map Attraction Over Time.html  # Static visualization of attraction over time
├── line chart+heat map.ipynb # Jupyter notebook with line charts and heatmaps
├── data/                     # Dataset folder
│   └── Speed Dating Data.csv # Raw data file
└── pages/                    # Dashboard pages
    └── 0_Overview_Dashboard.py # Overview dashboard page
```

## Dashboard Pages

### 1. Overview Dashboard (`0_Overview_Dashboard.py`)

Comprehensive dashboard featuring:

- **Decision Flow Sankey Diagram**: Visualizes the flow of dating decisions between participants
- **Persona Clustering Scatter Plot**: Shows participant personality clusters based on self-assessed traits
- **Dating Network Visualization**: Interactive network graph showing connection patterns between participants

### 2. Triangle Interaction Plot (`Triangle Interaction Plot.html`)

Static HTML visualization showing the relationship between three key stages:
- Pre-attraction
- Interaction Quality 
- Final Decision

This triangular representation illustrates how initial attraction, conversation quality, and final choice are interconnected.

### 3. Interaction Trajectory Map: Attraction Over Time (`Interaction Trajectory Map Attraction Over Time.html`)

3D visualization tracking how attraction ratings evolve over different stages of interaction, comparing trajectories for matches vs non-matches.

### 4. Line Chart and Heat Map Analysis (`line chart+heat map.ipynb`)

Jupyter notebook providing detailed temporal analysis of attribute ratings throughout the speed dating process:

- **Multi-stage Analysis**: Examines how participants' ratings of key attributes (attraction, fun, intelligence, sincerity, ambition) evolve from initial expectations through conversation stages to final decisions
- **Line Charts**: Illustrate rating trends across three distinct phases - Initial Expectations (Stage 1), Conversation Development (Stages 2-5), and Final Judgments
- **Correlation Heat Maps**: Display relationships between different attributes at each stage, revealing which traits influence others during the dating process
- **Gender-based Comparisons**: Contrasts how men and women rate various attributes differently throughout the interaction
- **Match Outcome Analysis**: Compares attribute rating patterns between successful matches and unsuccessful interactions
