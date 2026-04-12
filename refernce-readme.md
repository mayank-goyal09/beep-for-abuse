<div align="center">

# 🌦️ WeatherLens AI — Multi-City LSTM Forecasting

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Outfit&weight=700&size=32&duration=3500&pause=1000&color=4FACFE&center=true&vCenter=true&multiline=true&width=900&height=100&lines=Deep+Learning+Weather+Predictions+🌡️;30-Day+History+→+7-Day+Forecast;LSTM+Neural+Network+%7C+4+Global+Cities)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-WeatherLens_AI-4facfe?style=for-the-badge&labelColor=0c1445)](https://multi-city-lstm-weather-forecast-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/WeatherLens-AI?style=for-the-badge&color=ffd700)](https://github.com/mayank-goyal09/WeatherLens-AI/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/WeatherLens-AI?style=for-the-badge&color=87ceeb)](https://github.com/mayank-goyal09/WeatherLens-AI/network)

<br/>

<img src="https://user-images.githubusercontent.com/74038190/213910845-af37a709-8995-40d6-be59-724526e3c3d7.gif" width="800" alt="Weather Animation Banner"/>

<br/>

### 🧠 **Harnessing LSTM Neural Networks to predict 168 hours of temperature** 

### **From Open-Meteo Historical Data → Real-Time 7-Day Forecasts** 🌍

</div>

---

## ⚡ **THE FORECAST AT A GLANCE**

<table>
<tr>
<td width="50%">

### 🎯 **What This Project Does**

This end-to-end **deep learning weather forecasting system** uses **LSTM (Long Short-Term Memory)** neural networks trained on historical hourly data from **Open-Meteo** to predict temperature for the next **7 days (168 hours)** across **4 major global cities**.

**The Complete Pipeline:**
- 📡 **Data Collection** → Open-Meteo hourly historical API
- 🔄 **Sequence Transformation** → Supervised learning sequences
- 🧠 **Model Training** → Multi-step LSTM in Keras/TensorFlow
- 📊 **Visualization** → Interactive Streamlit dashboard
- 🚀 **Deployment** → Production-ready on Streamlit Cloud

</td>
<td width="50%">

### ✨ **Key Highlights**

| Feature | Details |
|---------|---------|
| 🌡️ **Forecast Horizon** | 7 days (168 hourly predictions) |
| 📅 **Input Window** | 30 days (720 hours of history) |
| 🏙️ **Cities Covered** | Delhi, Mumbai, New York, Los Angeles |
| 🧪 **Model Type** | Stacked LSTM Neural Network |
| 📉 **Performance** | Low MSE/MAE, beats naive baseline |
| 🎨 **UI Design** | Premium glassmorphism aesthetic |
| 📱 **Responsive** | Mobile-friendly dashboard |
| ⚡ **Real-Time** | Live predictions on demand |

</td>
</tr>
</table>

---

## 🌍 **CITIES IN FOCUS**

<div align="center">

| 🇮🇳 **Delhi** | 🇮🇳 **Mumbai** | 🇺🇸 **New York** | 🇺🇸 **Los Angeles** |
|:-------------:|:--------------:|:----------------:|:-------------------:|
| *Capital of India* | *Financial Hub* | *The Big Apple* | *City of Angels* |
| Extreme seasons | Tropical climate | Continental weather | Mediterranean vibes |

</div>

---

## 🛠️ **TECHNOLOGY STACK**

<div align="center">

![Tech Stack](https://skillicons.dev/icons?i=python,tensorflow,github,vscode)

</div>

| **Category** | **Technologies** | **Purpose** |
|:------------:|:-----------------|:------------|
| 🐍 **Core Language** | Python 3.8+ | Primary development language |
| 🧠 **Deep Learning** | TensorFlow / Keras | LSTM model architecture |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn | Data manipulation & preprocessing |
| 🎨 **Frontend** | Streamlit | Interactive web application |
| 📈 **Visualization** | Plotly, Matplotlib | Dynamic charts & graphs |
| 🌐 **Data Source** | Open-Meteo API | Historical weather data |
| 🚀 **Deployment** | Streamlit Cloud | Production hosting |

---

## 🔬 **HOW THE LSTM MODEL WORKS**

```mermaid
graph LR
    A[🌐 Open-Meteo API] --> B[📥 Historical Data]
    B --> C[🔄 Sequence Creation]
    C --> D[📊 30-Day Windows]
    D --> E[🧠 LSTM Network]
    E --> F[🌡️ 7-Day Predictions]
    F --> G[📱 Streamlit Dashboard]
    
    style A fill:#4facfe,color:#fff
    style E fill:#f093fb,color:#fff
    style G fill:#00f2fe,color:#000
```

### **The Pipeline Breakdown:**

<table>
<tr>
<td>

#### 📡 **1. Data Collection**
Fetch hourly historical weather data from **Open-Meteo** including:
- Temperature (2m)
- Relative Humidity
- Pressure (MSL)
- Wind Speed (10m)

</td>
<td>

#### 🔄 **2. Sequence Engineering**
Transform raw data into supervised sequences:
- **Input**: Last 30 days (720 hours)
- **Output**: Next 7 days (168 hours)
- Sliding window approach

</td>
</tr>
<tr>
<td>

#### 🧠 **3. LSTM Architecture**
Multi-layer LSTM network with:
- Stacked LSTM layers
- Dropout regularization
- Dense output layer (168 neurons)
- Adam optimizer

</td>
<td>

#### 📊 **4. Inference & Display**
Real-time predictions served via:
- City-specific last 30-day window
- Temperature forecast in °C
- Interactive Plotly visualizations

</td>
</tr>
</table>

---

## 📈 **MODEL PERFORMANCE**

<div align="center">

### 🏆 **Beating the Baseline**

</div>

| **Metric** | **LSTM Model** | **Naive Baseline** | **Improvement** |
|:----------:|:--------------:|:------------------:|:---------------:|
| 📉 **MSE** | Low | Higher | ✅ Significantly Better |
| 📏 **MAE** | Low | Higher | ✅ Significantly Better |
| 🎯 **Day 1** | Accurate | Moderate | ✅ Outperforms |
| 📅 **Days 2-7** | Consistent | Degrades | ✅ Stable Performance |

> 💡 **The naive persistence baseline** simply predicts "tomorrow = today." Our LSTM captures complex temporal patterns that this baseline misses entirely.

---

## 🎨 **DASHBOARD EXPERIENCE**

<div align="center">

### ✨ **Premium UI with Glassmorphism Design**

</div>

<table>
<tr>
<td width="50%">

#### 🏠 **Home View**
- **Hero Section** with gradient animations
- **Feature Cards** showcasing capabilities
- **City Selector** dropdown
- **Generate Forecast** action button

</td>
<td width="50%">

#### 📊 **Results View**
- **7-Day Line Chart** (hourly temperatures)
- **Daily Temperature Range** bar chart
- **Detailed Forecast Table** with emojis
- **Product-style display** (hourly Day 1, 4-hour steps Days 2-7)

</td>
</tr>
</table>

### **🎯 Design Highlights:**

```
✨ Dark mode with vibrant gradients
✨ Glassmorphism card effects  
✨ Smooth animations & transitions
✨ Interactive Plotly charts
✨ Weather condition emojis
✨ Responsive mobile layout
```

---

## 📂 **PROJECT STRUCTURE**

```
🌦️ WeatherLens-AI/
│
├── 📊 app.py                              # Streamlit dashboard application
├── 🧠 main.ipynb                          # EDA, training & evaluation notebook
├── ⚙️ weather_api.py                      # Open-Meteo data fetching utilities
│
├── 🗂️ Data & Models
│   ├── weather_hourly_history_openmeteo.csv   # Historical training data
│   ├── best_lstm_weather.h5               # Trained model checkpoint
│   ├── weather_lstm_7day.h5               # Final model file
│   └── weather_lstm_7day.weights.h5       # Model weights
│
├── 📦 requirements.txt                    # Project dependencies
└── 📖 README.md                           # You are here! 🎉
```

---

## 🚀 **QUICK START GUIDE**

<div align="center">

![Quick Start](https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif)

</div>

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/WeatherLens-AI.git
cd WeatherLens-AI
```

### **Step 2: Create Virtual Environment** 🐍

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 4: Launch the Dashboard** 🎯

```bash
streamlit run app.py
```

### **Step 5: Open in Browser** 🌐

Navigate to: **`http://localhost:8501`**

> 🎉 **That's it!** Select a city, hit "Generate Forecast," and watch the magic!

---

## 🎮 **HOW TO USE**

<table>
<tr>
<td>

### 🔹 **Generate Predictions**

1. Open the Streamlit dashboard
2. Select a city from the dropdown
3. Click **"🚀 Generate Forecast"**
4. View the 7-day temperature prediction
5. Explore interactive charts

</td>
<td>

### 🔹 **Analyze Forecasts**

1. Check the **7-Day Line Chart** for trends
2. Review **Daily Temperature Range** bars
3. Scroll through **Detailed Forecast** table
4. Notice weather condition emojis (☀️🌤️🌧️)
5. Compare hourly vs 4-hour step displays

</td>
</tr>
</table>

---

## 📚 **SKILLS DEMONSTRATED**

<div align="center">

### **A Portfolio-Ready Deep Learning Project**

</div>

| **Category** | **Skills** |
|:-------------|:-----------|
| 🧠 **Deep Learning** | LSTM architecture design, multi-step sequence forecasting |
| 📊 **Time-Series Analysis** | Sliding window sequences, feature engineering |
| 🔧 **Data Engineering** | API data collection, preprocessing pipelines |
| 🐍 **Python Development** | Clean code, modular architecture |
| 🎨 **UI/UX Design** | Glassmorphism, responsive design, premium aesthetics |
| 📈 **Data Visualization** | Interactive Plotly charts, meaningful displays |
| 🚀 **MLOps** | Model saving/loading, Streamlit Cloud deployment |
| 📐 **Model Evaluation** | MSE/MAE metrics, baseline comparisons |

---

## 🔮 **FUTURE ENHANCEMENTS**

- [ ] 🌍 Add more cities worldwide
- [ ] 🌧️ Predict additional weather variables (humidity, wind, precipitation)
- [ ] 📊 Implement attention mechanisms (Transformer architecture)
- [ ] 📈 Add confidence intervals for predictions
- [ ] 🔄 Real-time model retraining pipeline
- [ ] 📱 Progressive Web App (PWA) version
- [ ] 🌐 Multi-language support
- [ ] 📍 User location-based predictions
- [ ] 🤖 Weather alerts and notifications

---

## 🤝 **CONTRIBUTING**

<div align="center">

![Contributing](https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif)

</div>

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE**

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit_Site-4facfe?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mayank-portfolio-delta.vercel.app/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

<br/>

**Mayank Goyal**  
📊 Data Analyst | 🧠 Deep Learning Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

</div>

---

## ⭐ **SHOW YOUR SUPPORT**

<div align="center">

Give a ⭐️ if this project helped you understand LSTM forecasting and inspired your weather prediction projects!

<br/>

### 🌦️ **Built with Deep Learning & ❤️ by Mayank Goyal**

*"Predicting tomorrow's weather, one LSTM layer at a time!"* 🧠🌡️

<br/>

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:4facfe,100:00f2fe&height=120&section=footer)

</div>
