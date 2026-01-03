import datetime

# --- CONFIGURATION ---
current_year = datetime.datetime.now().year
CFTC_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
SENTIMENT_URL = "https://fxssi.com/tools/current-ratio"

ASSET_MAP = {
    "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD", "fxssi": "EURUSD"},
    "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD", "fxssi": "GBPUSD"},
    "USD/JPY":   {"base": "JAPANESE YEN", "usd": "USD INDEX", "yf": "JPY=X", "myfx": "USDJPY", "fxssi": "USDJPY"},
    "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD", "fxssi": "XAUUSD"},
    "Silver":    {"base": "SILVER", "usd": "USD INDEX", "yf": "SI=F", "myfx": "XAGUSD", "fxssi": "XAGUSD"},
    "EUR/GBP":   {"base": "EURO FX", "usd": "BRITISH POUND", "yf": "EURGBP=X", "myfx": "EURGBP", "fxssi": "EURGBP"},
    "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI", "fxssi": "USOIL"},
    "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD", "fxssi": "BTCUSD"},
    "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500", "fxssi": "US500"}
}

# SCALABLE WATCHLIST FOR SCANNER
WATCHLIST = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "EURGBP"]

# --- CONSTANTS ---
AI_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
COLOR_PRIMARY = "#38BDF8"
COLOR_BG = "#0F172A"
COLOR_CARD = "#1E293B"
COLOR_TEXT = "#F8FAFC"
FONT_MAIN = "Outfit, sans-serif"

# --- STYLES ---
# Theme: Premium Fintech (Slate)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        background-color: #0F172A; /* Slate 900 */
        font-family: 'Outfit', sans-serif;
    }
    
    /* Headers - High Contrast */
    h1, h2, h3, h4, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC !important; /* Slate 50 - Almost White */
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p, li, .stMarkdown p {
        color: #CBD5E1 !important; /* Slate 300 */
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #38BDF8 !important; /* Sky 400 */
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem !important;
        color: #94A3B8 !important; /* Slate 400 */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricDelta"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    /* Cards */
    .data-card {
        background-color: #1E293B; /* Slate 800 */
        border: 1px solid #334155; /* Slate 700 */
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: #94A3B8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        background-color: transparent !important;
        border-bottom: 2px solid #38BDF8;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617; /* Slate 950 */
        border-right: 1px solid #1E293B;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-color: #38BDF8;
        background-image: linear-gradient(to right, #38BDF8, #818CF8);
    }
    
    hr {
        border-color: #334155;
    }
</style>
"""
