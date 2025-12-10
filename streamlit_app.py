import streamlit as st

# ==================== Page Config ====================
st.set_page_config(
    page_title="Speed Dating Dashboard",
    page_icon="💘",
    layout="wide"
)

# ==================== Main Page ====================
st.title("💘 Speed Dating Data Visualization")

st.markdown("""
## Welcome to the Speed Dating Analysis Dashboard

This dashboard provides comprehensive visualizations of the Speed Dating Experiment dataset,
helping you understand patterns in dating preferences, match outcomes, and participant characteristics.

### 📊 Available Pages

Navigate using the sidebar to explore different aspects of the data:

---

#### 1️⃣ Match Landscape Overview
**Understanding the macro landscape of speed dating outcomes**

- Global match rate gauges (mutual match, dec, dec_o)
- Gender-based Yes rate comparison
- Decision breakdown (mutual match, one-sided, both no)
- Interactive wave filtering

---

#### 2️⃣ Participants Portrait Map
**Who are the speed dating participants?**

- Dimensionality reduction (UMAP/PCA) visualization
- KMeans clustering for participant segmentation
- Preference radar charts by cluster
- Demographic analysis (age, gender, dating frequency)
- Interactive color/size encoding options

---

### 🎯 Key Questions Answered

1. **How rare is a mutual match?** → See Match Landscape
2. **Do men and women have different Yes rates?** → See Match Landscape
3. **What types of people participate in speed dating?** → See Participants Portrait
4. **Are there distinct preference patterns among participants?** → See Participants Portrait

---

### 📖 Data Source

The data comes from the **Speed Dating Experiment** conducted by Columbia Business School professors.
Each participant engaged in multiple 4-minute "speed dates" and indicated whether they'd like to see each person again.

**Key Fields:**
- `dec`: Your decision (1=Yes, 0=No)
- `dec_o`: Partner's decision about you
- `match`: Mutual match (both said Yes)
- `attr1_1`, `sinc1_1`, etc.: Partner preference weights
- Demographics: age, gender, race, career, education

---

👈 **Select a page from the sidebar to begin exploring!**
""")

# ==================== Footer ====================
st.markdown("---")
st.caption("Built with Streamlit | Data Visualization Dashboard for Speed Dating Analysis")
