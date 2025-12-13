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

## Dataset

The dashboard uses the Speed Dating Experiment dataset from Columbia Business School. The dataset includes information about participants' demographics, preferences, and match outcomes from speed dating events.

## Project Structure

```
speed_dating/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation
├── data/                     # Dataset folder
│   └── Speed Dating Data.csv # Raw data file
└── pages/                    # Dashboard pages
    └── 0_Overview_Dashboard.py # Overview dashboard page
```

## Dependencies

All required packages and their versions are listed in `requirements.txt`. The main dependencies include:

- streamlit: For building the web dashboard
- pandas: For data manipulation and analysis
- plotly: For interactive visualizations
- numpy: For numerical computations
- scikit-learn: For machine learning algorithms
- umap-learn: For dimensionality reduction