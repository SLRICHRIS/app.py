import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Haptic-AR Smart Beekeeping",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and subtitle
st.title("🐝 Haptic-AR Smart Beekeeping System")
st.subheader("Precision Hive Monitoring | Early Disease Detection | Sustainable Pollination")

# Intro section
st.header("🔍 Introduction")
st.markdown("""
Bees are vital to agriculture, responsible for pollinating nearly 75% of all food crops. Yet, traditional beekeeping faces major challenges like **Colony Collapse Disorder (CCD)**, diseases, climate stress, and outdated manual inspections.

This system presents a modern, tech-enabled solution integrating:
- **Haptic & Augmented Reality (AR) interfaces**
- **IoT and biosensor modules**
- **AI-driven pollination optimization**
- **Cloud + Edge computing**

Together, they provide non-invasive, real-time monitoring of hive health and boost pollination efficiency in farmlands.
""")

# Feature list
st.header("🔧 System Features")

st.markdown("""
### ✅ Real-Time IoT-Based Hive Monitoring  
- Tracks temperature, humidity, vibration, gas levels (CO₂, NH₃, VOCs)  
- Detects anomalies without opening the hive

### ✅ Biosensor-Integrated Disease Detection  
- Monitors bee gut markers (pH, glucose, protein)  
- Detects early-stage infections like *Nosema* or viral stress

### ✅ Haptic-AR Remote Inspection  
- Uses AR headset + haptic glove  
- Allows remote hive exploration without physical disturbance

### ✅ AI-Driven Pollination Optimization  
- Drones scan crops, AI maps pollination density  
- Suggests optimal hive relocation to boost yield (20–30%)

### ✅ Haptic-VR Training Simulator  
- Virtual training for new beekeepers  
- Reduces risk of harming bees during learning

### ✅ Cloud + Edge Architecture  
- Edge for real-time alerts  
- Cloud for long-term analytics and storage
""")

# Problem justification
st.header("🚨 Why This Innovation Matters")

st.markdown("""
Manual hive checks disturb bees, raise colony stress, and often result in:
- Worker bees **killing their queen** (supersedure)
- **Larvae rejection** due to unstable conditions
- **Hive abandonment** in extreme cases

With biosensors and smart alerts, early action prevents colony loss, reduces cost, and promotes bee-friendly farming.
""")

# Optional image (you can add your own JPG/PNG here)
# st.image("hive-system.jpg", caption="System Architecture of Smart Hive", use_column_width=True)

# Contact section
st.header("📬 Contact")

st.markdown("""
**S. L. Richris**  
B.Tech ECE, Amrita School of Engineering, Amritapuri  
📧 slrichris2005@gmail.com  
🌐 [LinkedIn](https://linkedin.com/in/slrichris)
""")

# Footer
st.markdown("---")
st.markdown("🧠 Built with Streamlit • Last updated July 2025")
