# 🦅 Forecaster Pro - AI Trading Dashboard

**Forecaster Pro** is an advanced, analyst-grade trading dashboard built with Python and Streamlit. It fuses Technical Analysis, Institutional Data (COT), Retail Sentiment (FXSSI), and Generative AI (Google Gemini) to provide a "War Room" experience for Forex & Commodity traders.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Forecaster+Pro+Dashboard)

## 🌟 Key Features

### 1. 🔍 Market Scanner
- **Real-Time Confluence Matrix**: Instantly scans 6 major assets (EURUSD, GBPUSD, Gold, etc.).
- **Multi-Factor Scoring**: Combines Technicals, Sentiment, and Seasonality into a single 0-10 score.
- **Visual Traffic Lights**: "Green/Red" pills for immediate signal recognition.

### 2. 🏛️ Deep Dive Analysis
- **Institutional COT Data**: Tracks "Smart Money" positioning (Z-Scores & Net Flow).
- **Retail Sentiment**: Contrarian indicators based on real-time retail positioning.
- **Seasonality Engine**: 10-year historical win-rates for the current month.
- **Smart Logic**: Automatically handles "Partial" vs "Full" signals.

### 3. 🤖 AI Analyst (Powered by Gemini)
- **Context-Aware Intelligence**: The AI "sees" your dashboard data.
- **Executive Reports**: Generates professional trade plans (Bias, Risks, Execution) with one click.
- **Dynamic Model Selection**: Automatically uses the best available Gemini model.

## 🚀 Installation

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/forecaster-pro.git
cd forecaster-pro
pip install -r requirements.txt
```

### 2. Install Playwright (For Sentiment Scraping)
```bash
playwright install
```

### 3. Configure Secrets
Create a file `.streamlit/secrets.toml` and add your API Key:
```toml
GEMINI_KEY = "your_api_key_here"
```

## 🖥️ Usage

Run the dashboard locally:
```bash
streamlit run src/main.py
```
Or use the included `TradingDashboard.bat` file on Windows.

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Altair, Plotly
- **Backend Data**: YFinance, Playwright (FXSSI), Pandas
- **AI**: Google Generative AI (Gemini 1.5)

---
*Disclaimer: This tool is for educational purposes only. Trading involves risk.*
