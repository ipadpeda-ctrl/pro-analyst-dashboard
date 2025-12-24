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

# Configurazione pagina "Wide" per sfruttare tutto lo spazio
st.set_page_config(page_title="Pro Trend Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# CSS MODERNO & PULITO (CLEAN DASHBOARD STYLE)
# ==============================================================================
st.markdown("""
<style>
    /* Import Font Moderno */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* CARD STYLING - Contenitori puliti */
    .metric-card {
        background-color: #1e2130; /* Colore scuro moderno, non nero assoluto */
        border: 1px solid #2e3445;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    
    .metric-title {
        color: #a0aab9;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .delta-pos { color: #00cc96; }
    .delta-neg { color: #ef553b; }

    /* BIAS HEADER */
    .bias-header {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: 800;
        font-size: 1.2rem;
        color: white;
        letter-spacing: 1px;
    }
    
    /* CHART CONTAINERS */
    .chart-box {
        background-color: #161925;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e3445;
        margin-bottom: 15px;
    }
    
    h3 { font-size: 1.1rem !important; font-weight: 600; color: #e1e4e8; }
    h4 { font-size: 1.0rem !important; color: #cfd4da; }
</style>
""", unsafe_allow_html=True)

# 🔗 COSTANTI
current_year = datetime.now().year
CFTC_URL = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
SENTIMENT_URL = "https://www.myfxbook.com/community/outlook"

# ==============================================================================
# HELPER FUNCTIONS (GRAFICI & DATI)
# ==============================================================================

def make_candle_chart(hist_df, sma=None):
    """Crea un grafico pulito Prezzo + SMA"""
    source = hist_df.reset_index()
    base = alt.Chart(source).encode(x=alt.X('Date:T', axis=alt.Axis(format='%d %b', title=None)))
    
    # Candele (regola if open < close)
    rule = base.mark_rule().encode(
        y='Low:Q', y2='High:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#00cc96"), alt.value("#ef553b"))
    )
    bar = base.mark_bar().encode(
        y='Open:Q', y2='Close:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#00cc96"), alt.value("#ef553b"))
    )
    
    charts = [rule, bar]
    
    # Aggiungi SMA se presente
    if sma is not None:
        line = base.mark_line(color='orange', strokeWidth=2).encode(y=alt.Y('SMA:Q', scale=alt.Scale(zero=False)))
        charts.append(line)
        
    return alt.layer(*charts).properties(height=350, title="Prezzo & Trend (Daily)").interactive()

# ==============================================================================
# MOTORI ANALISI (AGGIORNATI)
# ==============================================================================

@st.cache_data(ttl=300)
def get_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y") # 1 Anno di dati per il grafico
        if hist.empty: return None
        
        # Indicatori
        curr = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg = ((curr - prev) / prev) * 100
        
        # SMA 200
        hist['SMA'] = hist['Close'].rolling(200).mean()
        sma200 = hist['SMA'].iloc[-1]
        
        # RSI Wilder
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        rsi = hist['RSI'].iloc[-1]
        
        # MACD
        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        status = "BULLISH" if curr > sma200 else "BEARISH" if sma200 else "NEUTRAL"
        
        return {
            "price": curr, "change": chg, "trend": status, 
            "rsi": rsi, "sma200": sma200, 
            "macd_hist": macd_hist.iloc[-1],
            "history": hist # Ritorniamo il DF per il grafico
        }
    except: return None

@st.cache_data(ttl=300)
def get_currency_strength(pair_ticker):
    try:
        # Codice identico al precedente, manteniamo la logica
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

@st.cache_data(ttl=3600)
def get_sentiment_data(target_pair):
    # Logica Scraping Myfxbook (invariata)
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
                    return {"status": "OK", "short": s_pct, "long": l_pct}
            return {"status": "Err", "msg": "N/A"}
        return {"status": "Err", "msg": "HTTP"}
    except Exception as e: return {"status": "Err", "msg": str(e)}

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
            df["Net"] = df["Long"] - df["Short"]
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            return df, True
        return pd.DataFrame(), False
    except: return pd.DataFrame(), False

def analyze_cot_pro(df, asset_name):
    # Logica COT (Percentile + Z-Score)
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
    
    lookback = d.head(156).copy() # 3 Anni
    curr = lookback.iloc[0]["Net"]
    
    std = lookback["Net"].std()
    mean = lookback["Net"].mean()
    z_score = (curr - mean) / std if std != 0 else 0
    percentile = (lookback["Net"] < curr).mean() * 100
    
    return {
        "z_score": z_score, "percentile": percentile, "net": curr,
        "history": lookback.set_index("Date")["Net"]
    }

@st.cache_data(ttl=86400)
def get_seasonality_pro(ticker):
    # Logica Stagionalità (Mediana)
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
        
        avg_path = piv_norm.median(axis=1)
        avg_path_smooth = avg_path.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        return {"win_rate": win_rate, "chart": avg_path_smooth, "day": datetime.now().timetuple().tm_yday}
    except: return None

# ==============================================================================
# 🧠 LOGICA "CONFLUENCE SCORE"
# ==============================================================================
def calculate_bias(tech, cot, sent, seas):
    score = 0
    reasons = []
    
    # 1. Tecnica (Trend Follower)
    if tech['trend'] == "BULLISH": score += 1; reasons.append("Prezzo sopra SMA200")
    else: score -= 1; reasons.append("Prezzo sotto SMA200")
    
    if tech['macd_hist'] > 0: score += 0.5; reasons.append("Momentum (MACD) Positivo")
    else: score -= 0.5
    
    # 2. COT (Mean Reversion sugli estremi)
    if cot:
        if cot['percentile'] > 90: score += 1; reasons.append("Istituzionali Extreme Long")
        elif cot['percentile'] < 10: score -= 1; reasons.append("Istituzionali Extreme Short")
    
    # 3. Sentiment (Contrarian)
    if sent['status'] == "OK":
        if sent['long'] > 65: score -= 1; reasons.append("Retail Troppo Long (Segnale Short)")
        elif sent['short'] > 65: score += 1; reasons.append("Retail Troppo Short (Segnale Long)")
        
    # 4. Stagionalità
    if seas:
        if seas['win_rate'] > 60: score += 0.5; reasons.append("Stagionalità Mese Positiva")
        elif seas['win_rate'] < 40: score -= 0.5
        
    final_bias = "NEUTRO / LATERALE ➡️"
    color = "#7f8c8d" # Grigio
    if score >= 1.5: 
        final_bias = "BULLISH (RIALZISTA) 🟢"
        color = "#00cc96"
    elif score <= -1.5: 
        final_bias = "BEARISH (RIBASSISTA) 🔴"
        color = "#ef553b"
        
    return final_bias, color, reasons

# ==============================================================================
# DASHBOARD LAYOUT
# ==============================================================================
df_cot, cot_ok = download_cot_data(CFTC_URL)

with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=43590&format=png", width=50) # Placeholder logo
    st.header("Settings")
    asset_map = {
        "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD"},
        "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD"},
        "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD"}, 
        "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI"}, 
        "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD"},
        "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500"}
    }
    sel_asset = st.selectbox("Seleziona Asset", list(asset_map.keys()))
    cfg = asset_map[sel_asset]
    
    st.markdown("---")
    # AI Key Section
    gemini_key = st.secrets["GEMINI_KEY"] if "GEMINI_KEY" in st.secrets else st.text_input("Gemini API Key", type="password")
    
    st.info("💡 **Consiglio:** Usa il layout a sinistra per l'analisi tecnica e a destra per i fondamentali.")

# --- FETCH DATA ---
tech = get_technical_analysis(cfg["yf"])
pwr = get_currency_strength(cfg["yf"])
seas = get_seasonality_pro(cfg["yf"])
sent = get_sentiment_data(cfg["myfx"])
cot_base = analyze_cot_pro(df_cot, cfg["base"]) if cot_ok else None

# --- HEADER: METRICHE CHIAVE ---
if tech:
    st.markdown(f"## {sel_asset}")
    c1, c2, c3, c4 = st.columns(4)
    
    # Custom HTML Metrics
    def metric_html(label, value, delta=None, color_delta=True):
        delta_html = ""
        if delta:
            color = "delta-pos" if ("+" in delta or float(delta.strip('%')) > 0) else "delta-neg"
            delta_html = f'<span class="metric-delta {color}">{delta}</span>'
        return f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """
        
    c1.markdown(metric_html("Prezzo", f"{tech['price']:.4f}", f"{tech['change']:+.2f}%"), unsafe_allow_html=True)
    c2.markdown(metric_html("RSI (14)", f"{tech['rsi']:.1f}", None), unsafe_allow_html=True)
    c3.markdown(metric_html("COT Percentile", f"{cot_base['percentile']:.0f}%" if cot_base else "N/A", None), unsafe_allow_html=True)
    
    sent_val = f"{sent['long']}% Long" if sent['status'] == 'OK' else "N/A"
    c4.markdown(metric_html("Sentiment Retail", sent_val, None), unsafe_allow_html=True)

# --- BIAS SCORE ---
if tech:
    bias, color, reasons = calculate_bias(tech, cot_base, sent, seas)
    st.markdown(f"""
    <div style="background-color: {color};" class="bias-header">
        STRATEGY BIAS: {bias}
    </div>
    """, unsafe_allow_html=True)

# --- MAIN GRID LAYOUT ---
col_main, col_side = st.columns([2, 1])

with col_main:
    # 1. GRAFICO PREZZO (Main Stage)
    st.markdown("### 📉 Analisi Tecnica")
    if tech:
        chart = make_candle_chart(tech['history'], sma=tech['sma200'])
        st.altair_chart(chart, use_container_width=True)
        
        # 2. INDICATORI SECONDARI (RSI/MACD)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.info(f"**MACD Hist:** {tech['macd_hist']:.4f}")
            st.progress(0.5 + (tech['macd_hist']*10)) # Visualizzazione rapida momentum
        with mc2:
            st.info(f"**Trend Lungo (SMA200):** {tech['trend']}")

    # 3. AI CHAT INTEGRATA (Sotto il grafico principale)
    st.markdown("### 💬 AI Analyst Insight")
    if gemini_key:
        user_input = st.chat_input("Chiedi un parere su questo grafico...")
        if user_input:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Context builder
            ctx = f"Asset: {sel_asset}, Prezzo: {tech['price']}, Bias Calcolato: {bias}, Motivi: {', '.join(reasons)}"
            prompt = f"Sei un trader esperto. Analizza brevemente: {ctx}. Domanda utente: {user_input}"
            
            with st.spinner("L'AI sta analizzando..."):
                try:
                    resp = model.generate_content(prompt)
                    st.success(resp.text)
                except Exception as e: st.error(str(e))
    else:
        st.warning("Inserisci API Key nella sidebar per attivare l'AI.")

with col_side:
    # DATI FONDAMENTALI (Stacked Cards)
    st.markdown("### 🧭 Contesto Macro")
    
    # Card 1: Power Meter
    if pwr:
        with st.container():
            st.markdown(f"**Forza Relativa (vs USD)**")
            asset_n = sel_asset.split('/')[0]
            st.progress(pwr['base']/10)
            st.caption(f"{asset_n}: {pwr['base']:.1f}/10")
            st.progress(pwr['usd']/10)
            st.caption(f"USD: {pwr['usd']:.1f}/10")
            st.markdown("---")

    # Card 2: COT Data
    if cot_base:
        st.markdown("**🏦 Posizionamento Istituzionale (COT)**")
        st.bar_chart(cot_base['history'].tail(52), height=120, color="#29b5e8")
        st.caption(f"Z-Score: {cot_base['z_score']:.2f} (Deviazioni dalla media)")
        st.markdown("---")

    # Card 3: Sentiment
    if sent['status'] == "OK":
        st.markdown("**👥 Sentiment Retail (Contrarian)**")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Short", f"{sent['short']}%", delta_color="inverse")
        col_s2.metric("Long", f"{sent['long']}%", delta_color="inverse")
        
        if sent['long'] > 70: st.error("⚠️ Retail troppo Long! Possibile calo.")
        elif sent['short'] > 70: st.success("⚠️ Retail troppo Short! Possibile rally.")
        st.markdown("---")

    # Card 4: Stagionalità
    if seas:
        st.markdown("**📅 Stagionalità (Mese Corrente)**")
        st.line_chart(seas['chart'], height=100)
        st.caption(f"Win Rate Storico: {int(seas['win_rate'])}%")