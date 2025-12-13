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

#### 1️⃣ Match  Overview Dashboard
**Understanding the macro landscape of speed dating outcomes**

- decision flow
- portrait clustering
- dating network

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

👈 **Select the Overview Dashboard from the sidebar to begin exploring!**
""")

# ==================== Footer ====================
st.markdown("---")
st.caption("Built with Streamlit | Data Visualization Dashboard for Speed Dating Analysis")