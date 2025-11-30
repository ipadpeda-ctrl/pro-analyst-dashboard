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

st.set_page_config(page_title="Forecaster Terminal Pro", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# CSS AVANZATO: STILE "FINANCIAL TERMINAL" (FONTS & CARDS)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@400;500;700&display=swap');

    /* SFONDO E FONT GLOBALI */
    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto', sans-serif;
    }
    
    /* TITOLI */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #e6e6e6;
    }
    
    /* METRICHE (NUMERI GRANDI) */
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Roboto', sans-serif;
        font-size: 0.85rem !important;
        color: #8b92a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* CARDS (RIQUADRI DATI) */
    .data-card {
        background-color: #161920;
        border: 1px solid #2d3139;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* BOX SEGNALI */
    .signal-box {
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        font-family: 'Roboto Mono', monospace;
        margin-top: 10px;
        text-transform: uppercase;
    }
    .bullish {background-color: #0f392b; color: #4caf50; border: 1px solid #1b5e20;}
    .bearish {background-color: #3e1a1a; color: #ff5252; border: 1px solid #b71c1c;}
    .neutral {background-color: #2d3139; color: #ffc107; border: 1px solid #856404;}

    /* PROGRESS BAR CUSTOM */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4caf50, #ff5252);
    }
    
    /* CHAT STYLES */
    .stChatMessage {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
    }
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
        Sei un Senior Hedge Fund Trader su un Terminale Finanziario.
        Analizza i dati per {context_data['asset']}:
        
        1. COT: Z-Score Asset {context_data['cot_z']:.2f} | USD {context_data['usd_z']:.2f}
        2. SENTIMENT: {context_data['sent_long']}% Long / {context_data['sent_short']}% Short
        3. STAGIONALITÀ: Win {context_data['seas_win']}% | Trend: {context_data['seas_trend']}
        4. TECNICA: Prezzo {context_data['price']} | Trend: {context_data['trend']} | RSI: {context_data['rsi']:.1f}
        5. FORZA RELATIVA: Asset {context_data['pwr_base']}/10 vs USD {context_data['pwr_usd']}/10
        
        Sii estremamente sintetico, professionale e diretto (stile Bloomberg). Usa elenchi puntati.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "🚦 Quota API superata. Attendi..."
        return f"Errore AI: {str(e)}"

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
        if sma200: status = "BULLISH" if curr > sma200 else "BEARISH"
            
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
    
    return {
        "name": best, "net": curr, "index": idx, "z_score": z_score, 
        "history": lookback.set_index("Date")["Net"], 
        "history_full": lookback.set_index("Date")[["Net", "Long", "Short", "OpenInt"]],
        "raw_data": d
    }

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
        avg_ret = hist_curr_m.mean() * 100 if len(hist_curr_m) > 0 else 0
        
        df_seas = hist[['Close']].copy()
        df_seas['DayOfYear'] = df_seas.index.dayofyear
        df_seas['Year'] = df_seas.index.year
        piv = df_seas.pivot_table(index='DayOfYear', columns='Year', values='Close')
        piv_norm = piv.apply(lambda x: (x / x.dropna().iloc[0]) * 100 if not x.dropna().empty else np.nan)
        avg_path = piv_norm.mean(axis=1)
        avg_path_smooth = avg_path.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        return {"win_rate": win_rate, "avg_return": avg_ret, "chart": avg_path_smooth, "day": datetime.now().timetuple().tm_yday}
    except: return None

# ==============================================================================
# DASHBOARD START
# ==============================================================================
st.title("FORECASTER TERMINAL PRO")

df_cot, cot_ok = download_cot_data(CFTC_URL)

# --- CONFIGURAZIONE ---
with st.sidebar:
    st.header("⚙️ SETTINGS")
    asset_map = {
        "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD"},
        "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD"},
        "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD"}, 
        "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI"}, 
        "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD"},
        "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500"}
    }
    sel_asset = st.selectbox("ASSET CLASS", list(asset_map.keys()))
    cfg = asset_map[sel_asset]
    
    st.markdown("---")
    if "GEMINI_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_KEY"]
        st.success("🔑 AI ONLINE")
    else:
        gemini_key = st.text_input("GEMINI API KEY", type="password")
        
    if gemini_key:
        st.caption("Model: Gemini 1.5 Flash (Fast)")
        
    # GUIDA RAPIDA SIDEBAR
    with st.expander("📘 LEGEND"):
        st.caption("**COT:** Z>2 Overbought, Z<-2 Oversold.")
        st.caption("**SENTIMENT:** >60% = Contrarian.")
        st.caption("**POWER:** >7 Strong, <3 Weak.")

# --- CALCOLI ---
tech = get_technical_analysis(cfg["yf"])
pwr = get_currency_strength(cfg["yf"])
seas = get_seasonality_pro(cfg["yf"])
sent = get_sentiment_data(cfg["myfx"])
cot_base, cot_usd = None, None
if cot_ok:
    cot_base = analyze_cot_pro(df_cot, cfg["base"])
    cot_usd = analyze_cot_pro(df_cot, cfg["usd"])

# --- ROW 1: TICKER HEADER ---
if tech:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRICE", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
    
    rsi_c = "inverse" if tech['rsi'] > 70 else "normal"
    c2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta_color=rsi_c)
    
    trend_col = "normal" if "BULL" in tech['trend_desc'] else "inverse"
    c3.metric("TREND (MA200)", tech['trend_desc'], delta_color=trend_col)
    
    c4.metric("VOLATILITY (ATR)", f"{tech['atr']:.4f}")
st.markdown("---")

# --- ROW 2: POWER METER ---
if pwr:
    c1, c2 = st.columns(2)
    asset_name = sel_asset.split('/')[0]
    with c1:
        st.write(f"**{asset_name} POWER:** {pwr['base']:.1f}/10")
        st.progress(pwr['base'] / 10)
    with c2:
        st.write(f"**USD POWER:** {pwr['usd']:.1f}/10")
        st.progress(pwr['usd'] / 10)
    st.markdown("---")

# --- ROW 3: MAIN GRID ---
c_cot, c_seas, c_sent = st.columns(3)

# CARD 1: COT
with c_cot:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("### 🏛️ MACRO COT")
    if cot_base and cot_usd:
        cl, cr = st.columns(2)
        cl.metric("Asset Z", f"{cot_base['z_score']:.2f}")
        cr.metric("USD Z", f"{cot_usd['z_score']:.2f}")
        
        # Mini Chart
        base_df = cot_base['history'].reset_index()
        chart = alt.Chart(base_df).mark_area(
            line={'color':'#29b5e8'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#29b5e8', offset=0),
                       alt.GradientStop(color='rgba(41, 181, 232, 0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(x=alt.X('Date', axis=None), y=alt.Y('Net', axis=None)).properties(height=80)
        st.altair_chart(chart, use_container_width=True)

        # Signal Logic
        if cot_base['z_score'] > 0.5 and cot_usd['z_score'] < -0.5:
            st.markdown('<div class="signal-box bullish">STRONG LONG</div>', unsafe_allow_html=True)
        elif cot_base['z_score'] < -0.5 and cot_usd['z_score'] > 0.5:
            st.markdown('<div class="signal-box bearish">STRONG SHORT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box neutral">NEUTRAL</div>', unsafe_allow_html=True)
    else: st.warning("No Data")
    st.markdown('</div>', unsafe_allow_html=True)

# CARD 2: SEASONALITY
with c_seas:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("### 📅 SEASONALITY")
    if seas:
        cl, cr = st.columns(2)
        cl.metric("Win Rate", f"{int(seas['win_rate'])}%")
        cr.metric("Avg Ret", f"{seas['avg_return']:.2f}%")
        
        # Mini Chart
        chart_df = pd.DataFrame({'Day': seas['chart'].index, 'Val': seas['chart'].values})
        line = alt.Chart(chart_df).mark_line(color='#d4edda').encode(
            x=alt.X('Day', axis=None), y=alt.Y('Val', scale=alt.Scale(zero=False), axis=None)
        ).properties(height=80)
        st.altair_chart(line, use_container_width=True)

        # Pendenza
        curr_day = seas['day']
        try:
            t_val = seas['chart'].iloc[curr_day-1]
            f_val = seas['chart'].iloc[min(curr_day+30, 364)]
            if f_val > t_val*1.005: st.markdown('<div class="signal-box bullish">CYCLE UP ↗️</div>', unsafe_allow_html=True)
            elif f_val < t_val*0.995: st.markdown('<div class="signal-box bearish">CYCLE DOWN ↘️</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="signal-box neutral">FLAT ➡️</div>', unsafe_allow_html=True)
            seas_trend = "UP" if f_val > t_val else "DOWN"
        except: 
            st.markdown('<div class="signal-box neutral">Calculating...</div>', unsafe_allow_html=True)
            seas_trend = "ND"
    else: st.write("Loading...")
    st.markdown('</div>', unsafe_allow_html=True)

# CARD 3: SENTIMENT
with c_sent:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("### 🐑 RETAIL SENTIMENT")
    if sent["status"] == "OK":
        cl, cr = st.columns(2)
        cl.metric("Shorts", f"{sent['short']}%")
        cr.metric("Longs", f"{sent['long']}%")
        
        st.progress(sent['long']/100)
        st.caption(f"Vol: {sent['vol']}")
        
        if sent['long'] > 60:
            st.markdown('<div class="signal-box bearish">CROWD LONG (SELL)</div>', unsafe_allow_html=True)
        elif sent['short'] > 60:
            st.markdown('<div class="signal-box bullish">CROWD SHORT (BUY)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box neutral">BALANCED</div>', unsafe_allow_html=True)
    else: st.error("No Data")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- DEEP DIVE & AI ---
t1, t2 = st.tabs(["📊 DEEP DIVE CHARTS", "🤖 AI ANALYST"])

with t1:
    if cot_base:
        st.markdown(f"#### {sel_asset} Positioning Structure")
        hist_data = cot_base['history_full'].reset_index()
        melted = hist_data.melt('Date', value_vars=['Long', 'Short'], var_name='Type', value_name='Contracts')
        chart_ls = alt.Chart(melted).mark_line().encode(
            x='Date', y='Contracts', color=alt.Color('Type', scale=alt.Scale(range=['green', 'red']))
        ).properties(height=300)
        st.altair_chart(chart_ls, use_container_width=True)
        
        st.markdown("#### Weekly Flows")
        raw = cot_base['raw_data'].head(3).copy()
        raw["Δ Long"] = raw["Long"] - raw["Long"].shift(-1)
        raw["Δ Short"] = raw["Short"] - raw["Short"].shift(-1)
        raw["Δ Net"] = raw["Net"] - raw["Net"].shift(-1)
        st.dataframe(raw.head(2)[["Date", "Long", "Δ Long", "Short", "Δ Short"]], use_container_width=True)

with t2:
    if gemini_key:
        if "messages" not in st.session_state: st.session_state.messages = []
        
        c_p1, c_p2, c_p3 = st.columns(3)
        prompt_btn = None
        if c_p1.button("📊 Report"): prompt_btn = f"Report completo su {sel_asset}."
        if c_p2.button("⚖️ COT Deep Dive"): prompt_btn = "Analizza posizionamento COT."
        if c_p3.button("🚦 Signal Check"): prompt_btn = "Consiglio operativo Buy/Sell."
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        user_input = st.chat_input("Ask AI...")
        if prompt_btn: user_input = prompt_btn
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)
            
            # CONTEXT
            ctx = {
                'asset': sel_asset,
                'cot_z': cot_base['z_score'] if cot_base else 0,
                'usd_z': cot_usd['z_score'] if cot_usd else 0,
                'pwr_base': pwr['base'] if pwr else 0, 'pwr_usd': pwr['usd'] if pwr else 0,
                'sent_long': sent['long'] if sent['status']=='OK' else 50,
                'sent_short': sent['short'] if sent['status']=='OK' else 50,
                'seas_win': int(seas['win_rate']) if seas else 50,
                'seas_trend': seas_trend if 'seas_trend' in locals() else "ND",
                'price': tech['price'], 'trend': tech['trend_desc'], 'rsi': tech['rsi']
            }
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = get_ai_analysis(gemini_key, "gemini-1.5-flash", ctx)
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
    else:
        st.info("Enter API Key in Sidebar to unlock AI.")