import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io
import requests
from bs4 import BeautifulSoup
import altair as alt
import google.generativeai as genai

st.set_page_config(page_title="Forecaster Terminal", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# CSS AVANZATO: STILE "FINANCIAL TERMINAL"
# ==============================================================================
st.markdown("""
<style>
    /* Sfondo generale e font */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Rimuove spazi inutili in alto */
    div.block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
    
    /* STILE DELLE CARD (I RIQUADRI) */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Metriche personalizzate */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-family: 'Roboto Mono', monospace;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #8b92a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Header Sezioni */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Box Segnali */
    .signal-box {
        padding: 10px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        font-size: 1rem;
        font-family: 'Roboto Mono', monospace;
        margin-top: 5px;
    }
    .bullish {background-color: #0f392b; color: #4caf50; border: 1px solid #1b5e20;}
    .bearish {background-color: #3e1a1a; color: #ff5252; border: 1px solid #b71c1c;}
    .neutral {background-color: #3e3518; color: #ffc107; border: 1px solid #856404;}

    /* Separatori */
    hr {margin-top: 1em; margin-bottom: 1em; border-color: #2d3139;}
    
</style>
""", unsafe_allow_html=True)

# 🔗 CONFIGURAZIONE
current_year = datetime.now().year
CFTC_URL = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
SENTIMENT_URL = "https://www.myfxbook.com/community/outlook"

# --- MOTORE AI ---
def get_filtered_models(api_key):
    try:
        genai.configure(api_key=api_key)
        priority_list = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro']
        all_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        available = [m for m in priority_list if m in all_models]
        if not available: available = [m for m in all_models if 'exp' not in m]
        return available
    except: return []

def get_ai_analysis(api_key, model_name, context_data):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        Sei un Senior Hedge Fund Trader. Analisi flash per {context_data['asset']}:
        DATI: COT Z-Score {context_data['cot_z']:.2f}, Forza Rel {context_data['pwr_base']:.1f}/10, Sentiment Retail {context_data['sent_long']}% Long, Stagionalità Win {context_data['seas_win']}%, Prezzo {context_data['price']}, RSI {context_data['rsi']:.1f}.
        
        Rispondi in stile "Terminal Bloomberg" (sintetico, bullet points):
        1. Bias Macro
        2. Rischi
        3. Action (BUY/SELL/WAIT)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "🚦 Quota superata."
        return f"Err: {str(e)}"

# --- MOTORE 1: TECNICA ---
def get_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        curr = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg = ((curr - prev) / prev) * 100
        sma200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        sma50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        high_low = hist["High"] - hist["Low"]
        h_c = np.abs(hist["High"] - hist["Close"].shift())
        l_c = np.abs(hist["Low"] - hist["Close"].shift())
        tr = np.max(pd.concat([high_low, h_c, l_c], axis=1), axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        status = "N/A"
        if sma200: status = "BULL" if curr > sma200 else "BEAR"
            
        return {"price": curr, "change": chg, "trend_desc": status, "rsi": rsi, "atr": atr, "sma200": sma200}
    except: return None

# --- MOTORE 1.5: FORZA RELATIVA ---
def get_currency_strength(pair_ticker):
    try:
        tickers = f"DX-Y.NYB {pair_ticker}"
        data = yf.download(tickers, period="1mo", progress=False)['Close']
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            dxy = data["DX-Y.NYB"]
            pair = data[pair_ticker]
        else:
            dxy = data["DX-Y.NYB"]
            pair = data[pair_ticker]

        dxy_move = dxy.pct_change(5).iloc[-1] * 100
        pair_move = pair.pct_change(5).iloc[-1] * 100
        
        usd_score = 5 + (dxy_move * 2.0)
        base_chg_est = pair_move + dxy_move
        base_score = 5 + (base_chg_est * 2.0)
        return {"base": max(0, min(10, base_score)), "usd": max(0, min(10, usd_score))}
    except: return None

# --- MOTORE 2: SENTIMENT ---
@st.cache_data(ttl=3600)
def get_sentiment_data(target_pair):
    headers = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
    try:
        r = requests.get(SENTIMENT_URL, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            cell = soup.find(lambda tag: tag.name == "td" and target_pair in tag.text)
            if cell:
                row = cell.parent
                s_cell = row.find("td", string="Short")
                l_cell = row.find("td", string="Long")
                if s_cell and l_cell:
                    s_pct = int(s_cell.find_next_sibling("td").text.strip().replace("%", ""))
                    l_pct = int(l_cell.find_next_sibling("td").text.strip().replace("%", ""))
                    vol = s_cell.find_next_sibling("td").find_next_sibling("td").text.strip()
                    return {"status": "OK", "short": s_pct, "long": l_pct, "vol": vol}
            return {"status": "Err", "msg": "Pair not found"}
        return {"status": "Err", "msg": "HTTP Err"}
    except Exception as e: return {"status": "Err", "msg": str(e)}

# --- MOTORE 3: COT ---
@st.cache_data(ttl=86400)
def download_cot_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        if r.status_code == 200:
            with io.BytesIO(r.content) as z:
                df = pd.read_csv(z, header=None, compression='zip', usecols=[0, 2, 7, 8, 9], low_memory=False)
            df.columns = ["Asset", "Date", "OpenInt", "Long", "Short"]
            df["Asset"] = df["Asset"].str.strip().str.upper()
            df["Long"] = pd.to_numeric(df["Long"], errors='coerce').fillna(0)
            df["Short"] = pd.to_numeric(df["Short"], errors='coerce').fillna(0)
            df["OpenInt"] = pd.to_numeric(df["OpenInt"], errors='coerce').fillna(0)
            df["Net"] = df["Long"] - df["Short"]
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            return df, True
        return pd.DataFrame(), False
    except: return pd.DataFrame(), False

def analyze_cot_pro(df, asset_name):
    mask = df["Asset"].str.contains(asset_name, case=False, na=False)
    if asset_name in ["EURO FX", "BRITISH POUND"]: 
        exclude_mask = df["Asset"].str.contains("XRATE", case=False, na=False)
        mask = mask & ~exclude_mask 
    matches = df[mask]["Asset"].unique()
    if len(matches) == 0: return None
    
    if "USD" in asset_name and any("ICE" in m for m in matches):
        best = [m for m in matches if "ICE" in m][0]
    else:
        best = sorted(matches, key=len)[0]
        
    d = df[df["Asset"] == best].sort_values("Date", ascending=False).drop_duplicates(subset=['Date'])
    if len(d) < 2: return None
    
    lookback = d.head(52).copy()
    curr = lookback.iloc[0]["Net"]
    std = lookback["Net"].std()
    mean = lookback["Net"].mean()
    idx = ((curr - lookback["Net"].min()) / (lookback["Net"].max() - lookback["Net"].min())) * 100
    z_score = (curr - mean) / std if std != 0 else 0
    
    return {"name": best, "net": curr, "index": idx, "z_score": z_score, "history": lookback.set_index("Date")["Net"], "history_full": lookback.set_index("Date")[["Net", "Long", "Short", "OpenInt"]], "raw_data": d}

# --- MOTORE 4: STAGIONALITÀ ---
@st.cache_data(ttl=86400)
def get_seasonality_pro(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y")
        if hist.empty or len(hist) < 200: return None
        m_ret = hist['Close'].resample('ME').ffill().pct_change().dropna()
        curr_m = datetime.now().month
        hist_curr_m = m_ret[m_ret.index.month == curr_m].dropna()
        win_rate = (len(hist_curr_m[hist_curr_m > 0]) / len(hist_curr_m)) * 100 if len(hist_curr_m) > 0 else 50
        
        df_seas = hist[['Close']].copy()
        df_seas['DayOfYear'] = df_seas.index.dayofyear
        df_seas['Year'] = df_seas.index.year
        piv = df_seas.pivot_table(index='DayOfYear', columns='Year', values='Close')
        piv_norm = piv.apply(lambda x: (x / x.dropna().iloc[0]) * 100 if not x.dropna().empty else np.nan)
        avg_path = piv_norm.mean(axis=1)
        avg_path_smooth = avg_path.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        return {"win_rate": win_rate, "chart": avg_path_smooth, "day": datetime.now().timetuple().tm_yday}
    except: return None

# ==============================================================================
# DASHBOARD LAYOUT
# ==============================================================================
st.title("TERMINAL ALPHA 1.0")

df_cot, cot_ok = download_cot_data(CFTC_URL)

# SIDEBAR (Minimale)
with st.sidebar:
    st.header("Asset Selection")
    asset_map = {
        "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD"},
        "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD"},
        "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD"}, 
        "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI"}, 
        "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD"},
        "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500"}
    }
    sel_asset = st.selectbox("Simbolo", list(asset_map.keys()))
    cfg = asset_map[sel_asset]
    
    st.markdown("---")
    if "GEMINI_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_KEY"]
        st.success("AI Online")
    else:
        gemini_key = st.text_input("API Key", type="password")
        
    # Pulsanti AI Sidebar
    prompt_btn = None
    if gemini_key:
        st.markdown("### AI Tools")
        if st.button("📋 Report Flash"): prompt_btn = "Report sintetico"
        if st.button("🚦 Segnale Operativo"): prompt_btn = "Dammi un segnale"

# CALCOLI
tech = get_technical_analysis(cfg["yf"])
pwr = get_currency_strength(cfg["yf"])
seas = get_seasonality_pro(cfg["yf"])
sent = get_sentiment_data(cfg["myfx"])
cot_base, cot_usd = None, None
if cot_ok:
    cot_base = analyze_cot_pro(df_cot, cfg["base"])
    cot_usd = analyze_cot_pro(df_cot, cfg["usd"])

# --- ROW 1: KPI HEADERS (TICKER STYLE) ---
if tech:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(f"{sel_asset}", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
    
    rsi_c = "off" # Neutro
    if tech['rsi'] > 70: rsi_c = "inverse" # Bear
    if tech['rsi'] < 30: rsi_c = "normal" # Bull
    kpi2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta_color=rsi_c)
    
    trend_col = "normal" if "BULL" in tech['trend_desc'] else "inverse"
    kpi3.metric("Trend (200MA)", tech['trend_desc'], delta_color=trend_col)
    
    kpi4.metric("Volatilità (ATR)", f"{tech['atr']:.4f}")

st.markdown("---")

# --- ROW 2: MAIN ENGINES (GRID LAYOUT) ---
c_cot, c_seas, c_sent = st.columns(3)

# 1. COT CARD
with c_cot:
    st.markdown("### 🏛️ ISTITUZIONALI (COT)")
    if cot_base and cot_usd:
        col_l, col_r = st.columns(2)
        col_l.metric("Asset Z", f"{cot_base['z_score']:.2f}")
        col_r.metric("USD Z", f"{cot_usd['z_score']:.2f}")
        
        # Signal Logic
        if cot_base['z_score'] > 0.5 and cot_usd['z_score'] < -0.5:
            st.markdown('<div class="signal-box bullish">STRONG LONG</div>', unsafe_allow_html=True)
        elif cot_base['z_score'] < -0.5 and cot_usd['z_score'] > 0.5:
            st.markdown('<div class="signal-box bearish">STRONG SHORT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box neutral">NEUTRALE</div>', unsafe_allow_html=True)
            
        st.area_chart(cot_base['history'], height=100, color="#29b5e8")
    else:
        st.warning("Dati COT Indisponibili")

# 2. SEASONALITY CARD
with c_seas:
    st.markdown("### 📅 STAGIONALITÀ")
    if seas:
        col_l, col_r = st.columns(2)
        col_l.metric("Win Rate", f"{int(seas['win_rate'])}%")
        col_r.metric("Ciclo", "UP" if seas['avg_return']>0 else "DOWN")
        
        # Signal Logic
        if seas['win_rate'] >= 65:
            st.markdown('<div class="signal-box bullish">BULLISH MONTH</div>', unsafe_allow_html=True)
        elif seas['win_rate'] <= 35:
            st.markdown('<div class="signal-box bearish">BEARISH MONTH</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box neutral">MISTO</div>', unsafe_allow_html=True)

        # Mini Chart
        chart_df = pd.DataFrame({'Giorno': seas['chart'].index, 'Valore': seas['chart'].values})
        line = alt.Chart(chart_df).mark_line(color='white').encode(
            x=alt.X('Giorno', axis=None), y=alt.Y('Valore', scale=alt.Scale(zero=False), axis=None)
        ).properties(height=100)
        st.altair_chart(line, use_container_width=True)
    else:
        st.write("Loading...")

# 3. SENTIMENT CARD
with c_sent:
    st.markdown("### 🐑 RETAIL SENTIMENT")
    if sent["status"] == "OK":
        # Calcolo forza relativa qui per visualizzazione compatta
        if pwr:
            st.caption(f"Strength: {sel_asset.split('/')[0]} {pwr['base']:.1f} | USD {pwr['usd']:.1f}")
        
        c_s, c_l = st.columns(2)
        c_s.metric("Short", f"{sent['short']}%")
        c_l.metric("Long", f"{sent['long']}%")
        
        if sent['long'] > 60:
            st.markdown('<div class="signal-box bearish">CROWD LONG (SELL)</div>', unsafe_allow_html=True)
        elif sent['short'] > 60:
            st.markdown('<div class="signal-box bullish">CROWD SHORT (BUY)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box neutral">NEUTRALE</div>', unsafe_allow_html=True)
            
        st.progress(sent['long']/100)
    else:
        st.error("No Data")

st.markdown("---")

# --- ROW 3: DETAILS & AI ---
t1, t2 = st.tabs(["📊 Dettagli & Grafici", "🤖 Chat Analyst"])

with t1:
    if cot_base:
        st.markdown("##### Struttura COT (Green=Long, Red=Short)")
        hist_data = cot_base['history_full'].reset_index()
        melted = hist_data.melt('Date', value_vars=['Long', 'Short'], var_name='Type', value_name='Contracts')
        chart_ls = alt.Chart(melted).mark_line().encode(
            x='Date', y='Contracts', color=alt.Color('Type', scale=alt.Scale(range=['green', 'red']))
        ).properties(height=300)
        st.altair_chart(chart_ls, use_container_width=True)

with t2:
    if gemini_key:
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        # Handle Button Prompt
        if prompt_btn:
            st.session_state.messages.append({"role": "user", "content": prompt_btn})
            with st.chat_message("user"): st.markdown(prompt_btn)
            
            ctx = {
                'asset': sel_asset, 'cot_z': cot_base['z_score'] if cot_base else 0,
                'pwr_base': pwr['base'] if pwr else 0, 'pwr_usd': pwr['usd'] if pwr else 0,
                'usd_z': cot_usd['z_score'] if cot_usd else 0,
                'sent_long': sent['long'] if sent['status']=='OK' else 50,
                'sent_short': sent['short'] if sent['status']=='OK' else 50,
                'seas_win': int(seas['win_rate']) if seas else 50, 'seas_trend': "ND",
                'price': tech['price'], 'trend': tech['trend_desc'], 'rsi': tech['rsi']
            }
            with st.chat_message("assistant"):
                 resp = get_ai_analysis(gemini_key, "gemini-1.5-flash", ctx)
                 st.markdown(resp)
                 st.session_state.messages.append({"role": "assistant", "content": resp})

        # Handle Manual Input
        if user_input := st.chat_input("Chiedi all'analista..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)
            ctx = {'asset': sel_asset, 'cot_z': cot_base['z_score'] if cot_base else 0, 'usd_z': cot_usd['z_score'] if cot_usd else 0, 'sent_long': sent['long'] if sent['status']=='OK' else 50, 'sent_short': sent['short'] if sent['status']=='OK' else 50, 'seas_win': int(seas['win_rate']) if seas else 50, 'seas_trend': "ND", 'price': tech['price'], 'trend': tech['trend_desc'], 'rsi': tech['rsi'], 'pwr_base': pwr['base'] if pwr else 0, 'pwr_usd': pwr['usd'] if pwr else 0}
            with st.chat_message("assistant"):
                resp = get_ai_analysis(gemini_key, "gemini-1.5-flash", ctx)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
    else:
        st.info("Inserisci la chiave API nella sidebar")