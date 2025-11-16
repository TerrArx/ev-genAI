# ⚡ EV Specification Generator

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evospex.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Design your perfect electric vehicle with AI-powered recommendations**

[🚀](https://evospex.streamlit.app/) | [📊](https://www.kaggle.com/datasets/pratyushpuri/ev-electrical-vehicles-dataset-3k-records-2025)
</div>

---

## 🎯 What is This?

This project uses **Conditional Variational Autoencoders (CVAE)** to generate realistic electric vehicle specifications based on user preferences. Simply choose your desired region, battery chemistry, charging type, and budget—and let the AI design an optimized EV configuration for you!

### ✨ Key Features

- 🎨 **Interactive Design Studio** - Customize your EV preferences in real-time
- 🤖 **AI-Powered Generation** - CVAE model trained on 3K+ real EV specifications
- 💬 **Smart Assistant** - Chat with Gemini AI about your generated specs
- 📊 **Performance Analytics** - Visualize efficiency, cost, and market comparisons
- 📄 **PDF Reports** - Download professional specification reports

---

## 🚀 Try It Now

**Live App:** [https://evospex.streamlit.app/](https://evospex.streamlit.app/)

### How to Use:
1. Select your preferences (region, battery type, connector, etc.)
2. Click "Generate EV Specification"
3. Explore the AI-generated specs and visualizations
4. Chat with the assistant for insights
5. Download a professional PDF report

---

## 🏗️ Architecture

The system uses a **Conditional Variational Autoencoder** with:

- **Encoder**: Maps vehicle specs + conditions → 12D latent space
- **Decoder**: Generates specs from latent space + user conditions
- **Training**: Beta annealing (0.05 → 1.0) over 150 epochs
- **Dataset**: 3K+ pure BEVs across 40+ countries, 4 battery chemistries

**Generated Features:**
- 🔋 Battery Capacity (kWh)
- 🛣️ Driving Range (km)
- 💵 Target Price (USD)

**User Conditions:**
- 🌍 Market Region (Europe, Asia, North America, Rest of World)
- 🔋 Battery Chemistry (Li-ion, LFP, Advanced, Legacy)
- 🔌 Charging Connector (CCS, NACS, CHAdeMO)
- ⚡ Charging Speed (Slow, Fast, Ultra-Fast)
- 💰 Budget Class (Budget, Mid-Range, Luxury)
- 🔄 V2X Technology (Yes/No)

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/TerrArx/ev-genAI.git
cd ev-genAI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Configuration

Create `.streamlit/secrets.toml` for Gemini API:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

---

## 🛠️ Project Structure

```
ev-genAI/
├── app.py                      # Main Streamlit application
├── style.css                   # Dark minimalist theme
├── ev-preprocess.py            # Data cleaning & feature engineering
├── ev-train.py                 # CVAE model training
├── cvae.py                     # Model loading utilities
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version (3.11)
├── data-encoded-ev/           # Preprocessed data & artifacts
│   ├── training_data.npz
│   ├── scaler_y.pkl
│   ├── encoder_c.pkl
│   └── budget_bins.json
├── models/keras format/       # Trained CVAE models
│   ├── encoder.keras
│   └── decoder.keras
└── dataset/                   # Original EV dataset
    └── electric_vehicles_dataset.csv
```

---

## 🎓 Educational Context

This project was developed as part of the:

**🌱 Edunet Foundation × Shell Skills4Future AICTE Internship**

**Focus Areas:**
- Green Skills & Sustainability
- Artificial Intelligence for Climate Action
- Electric Vehicle Innovation
- Generative AI Applications

---

## 📊 Dataset Information

**Source:** [EV Electrical Vehicles Dataset - Kaggle](https://www.kaggle.com/datasets/pratyushpuri/ev-electrical-vehicles-dataset-3k-records-2025)

**Content:**
- 3,000+ pure Battery Electric Vehicles (BEVs)
- 40+ countries consolidated into 4 regions
- Multiple battery chemistry families
- Real-world pricing and performance data
- Charging infrastructure specifications

---

## 🔬 Technical Details

### Model Training
- **Framework:** TensorFlow/Keras 3.x
- **Architecture:** Conditional VAE with 12D latent space
- **Loss Function:** Reconstruction + KL Divergence with beta annealing
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau
- **Training:** 150 epochs, batch size 64

### Tech Stack
- **Frontend:** Streamlit 1.39+
- **ML/AI:** TensorFlow, Keras, scikit-learn
- **Generative AI:** Google Gemini 2.5-flash
- **Visualization:** Plotly, Kaleido
- **PDF Generation:** ReportLab

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 👨‍💻 Author

**Nabil Ahmed**  
B.Tech in Artificial Intelligence and Machine Learning  
Netaji Subhash Engineering College, Kolkata, India

📧 Email: nabil13147@gmail.com  
🐙 GitHub: [@TerrArx](https://github.com/TerrArx)  
💼 LinkedIn: [Nabil Ahmed](https://www.linkedin.com/in/nabil-ahmed-876b30240/)

---

## ⚠️ Disclaimer

This is a **demonstration project** using synthetic AI-generated data for educational purposes. The specifications shown are **not real** and should **not** be used for actual vehicle purchasing decisions or commercial purposes.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ and ⚡ for a sustainable future

</div>
