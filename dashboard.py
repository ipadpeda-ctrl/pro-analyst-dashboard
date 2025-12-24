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

st.set_page_config(page_title="Pro Trend Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# CSS AVANZATO: STILE "FINANCIAL TERMINAL"
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@400;500;700&display=swap');

    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #e6e6e6;
    }
    
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

    .data-card {
        background-color: #161920;
        border: 1px solid #2d3139;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4caf50, #ff5252);
    }
    
    .stChatMessage {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
    }
    
    button[data-baseweb="tab"] {
        font-family: 'Roboto', sans-serif;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 🔗 CONFIGURAZIONE
current_year = datetime.now().year
CFTC_URL = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
SENTIMENT_URL = "https://www.myfxbook.com/community/outlook"

# ==============================================================================
# MOTORE AI (Google Gemini)
# ==============================================================================
def get_filtered_models(api_key):
    try:
        genai.configure(api_key=api_key)
        priority_list = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro']
        all_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        available = [m for m in priority_list if m in all_models]
        if not available: available = [m for m in all_models if 'exp' not in m]
        return available
    except: return []

# ==============================================================================
# MOTORE 1: TECNICA (RSI Wilder's Smoothing & ATR)
# ==============================================================================
def get_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Scarichiamo più dati per stabilizzare la media esponenziale
        hist = stock.history(period="2y")
        if hist.empty: return None
        curr = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg = ((curr - prev) / prev) * 100
        sma200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        sma50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        
        # Calcolo RSI Corretto (Wilder's Smoothing)
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # ewm con alpha=1/14 approssima il metodo Wilder originale
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR Calculation
        high_low = hist["High"] - hist["Low"]
        h_c = np.abs(hist["High"] - hist["Close"].shift())
        l_c = np.abs(hist["Low"] - hist["Close"].shift())
        tr = np.max(pd.concat([high_low, h_c, l_c], axis=1), axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        status = "NEUTRO"
        if sma200:
            status = "ALZISTA" if curr > sma200 else "RIBASSISTA"
            
        return {"price": curr, "change": chg, "trend_desc": status, "rsi": rsi, "atr": atr, "sma200": sma200}
    except: return None

# ==============================================================================
# MOTORE 1.5: FORZA RELATIVA
# ==============================================================================
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

# ==============================================================================
# MOTORE 2: SENTIMENT (Scraping Myfxbook)
# ==============================================================================
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

# ==============================================================================
# MOTORE 3: COT (Download & Analisi Percentile)
# ==============================================================================
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
    
    # Lookback esteso a 3 anni (156 settimane) per significatività statistica
    lookback_period = 156 if len(d) >= 156 else len(d)
    lookback = d.head(lookback_period).copy()
    
    curr = lookback.iloc[0]["Net"]
    
    # Z-Score
    std = lookback["Net"].std()
    mean = lookback["Net"].mean()
    z_score = (curr - mean) / std if std != 0 else 0
    
    # Percentile Rank (Più robusto)
    percentile = (lookback["Net"] < curr).mean() * 100
    
    idx = ((curr - lookback["Net"].min()) / (lookback["Net"].max() - lookback["Net"].min())) * 100 if (lookback["Net"].max() - lookback["Net"].min()) != 0 else 50
    
    return {
        "name": best, "net": curr, "index": idx, "z_score": z_score, "percentile": percentile,
        "history": lookback.set_index("Date")["Net"], 
        "history_full": lookback.set_index("Date")[["Net", "Long", "Short", "OpenInt"]],
        "raw_data": d
    }

# ==============================================================================
# MOTORE 4: STAGIONALITÀ (Media vs Mediana)
# ==============================================================================
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
        
        # USA LA MEDIANA (Più robusta agli outlier)
        avg_path = piv_norm.median(axis=1)
        avg_path_smooth = avg_path.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        return {"win_rate": win_rate, "chart": avg_path_smooth, "day": datetime.now().timetuple().tm_yday}
    except: return None

# ==============================================================================
# DASHBOARD START
# ==============================================================================
st.title("🦅 Forecaster Terminal Pro")

# --- GUIDA COMPLETA ---
with st.expander("📘 MANUALE OPERATIVO (Clicca per aprire)"):
    t1, t2, t3 = st.tabs(["🏦 Macro COT", "📅 Stagionalità", "🚦 Strategia"])
    with t1:
        st.markdown("""
        * **Z-Score:** Deviazione dalla media. > +2.0 (Ipercomprato), < -2.0 (Ipervenduto).
        * **Percentile:** 90% significa che oggi i net long sono superiori al 90% degli ultimi 3 anni.
        """)
    with t2:
        st.markdown("""
        * **Win Rate:** Probabilità storica di chiusura positiva per il mese corrente.
        * **Median Path:** Il grafico mostra l'anno "tipico", ignorando gli anni anomali (outlier).
        """)
    with t3:
        st.markdown("""
        * **Sentiment:** Se Retail Long > 60% -> Cerca Short (Contrarian).
        * **RSI:** Calcolato con metodo Wilder (standard industriale).
        """)

# --- CARICAMENTO DATI COT ---
df_cot, cot_ok = download_cot_data(CFTC_URL)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ CONFIGURAZIONE")
    asset_map = {
        "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD"},
        "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD"},
        "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD"}, 
        "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI"}, 
        "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD"},
        "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500"}
    }
    sel_asset = st.selectbox("Asset Class", list(asset_map.keys()))
    cfg = asset_map[sel_asset]
    
    st.markdown("---")
    if "GEMINI_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_KEY"]
        st.success("🔑 AI Ready")
    else:
        gemini_key = st.text_input("API Key (Opzionale)", type="password")
    
    selected_model = None
    if gemini_key:
        mods = get_filtered_models(gemini_key)
        if mods: selected_model = st.selectbox("Modello AI", mods)

# --- ESECUZIONE CALCOLI ---
tech = get_technical_analysis(cfg["yf"])
pwr = get_currency_strength(cfg["yf"])
seas = get_seasonality_pro(cfg["yf"])
sent = get_sentiment_data(cfg["myfx"])
cot_base, cot_usd = None, None
if cot_ok:
    cot_base = analyze_cot_pro(df_cot, cfg["base"])
    cot_usd = analyze_cot_pro(df_cot, cfg["usd"])

# --- TOP BAR ---
c1, c2, c3, c4 = st.columns(4)
if tech:
    c1.metric("PREZZO", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
    c2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta_color="off")
    c3.metric("TREND (MA200)", tech['trend_desc'], delta_color="off")
    c4.metric("VOLATILITÀ (ATR)", f"{tech['atr']:.4f}")

st.markdown("---")

# --- FORZA RELATIVA ---
if pwr:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("### 💪 FORZA RELATIVA (POWER METER)")
    cp1, cp2 = st.columns(2)
    asset_name = sel_asset.split('/')[0]
    with cp1:
        st.write(f"**{asset_name}**: {pwr['base']:.1f}/10")
        st.progress(pwr['base'] / 10)
    with cp2:
        st.write(f"**USD**: {pwr['usd']:.1f}/10")
        st.progress(pwr['usd'] / 10)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 MACRO COT", "📅 STAGIONALITÀ", "🐑 SENTIMENT"])

with tab1:
    if cot_base and cot_usd:
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"#### {sel_asset.split('/')[0]}")
            st.metric("Z-Score", f"{cot_base['z_score']:.2f}", help="Standard Deviation")
            st.metric("Percentile Rank", f"{cot_base['percentile']:.1f}%", help="Posizionamento rispetto ultimi 3 anni")
            st.area_chart(cot_base['history'], height=150, color="#29b5e8")
        with c_b:
            st.markdown("#### USD (Dollaro)")
            st.metric("Z-Score", f"{cot_usd['z_score']:.2f}")
            st.metric("Percentile Rank", f"{cot_usd['percentile']:.1f}%")
            st.area_chart(cot_usd['history'], height=150, color="#ffaa00")
            
        with st.expander("🔬 Analisi Approfondita (Long/Short & OI)"):
            st.markdown(f"##### {sel_asset}: Long vs Short")
            hist_data = cot_base['history_full'].reset_index()
            melted = hist_data.melt('Date', value_vars=['Long', 'Short'], var_name='Type', value_name='Contracts')
            chart_ls = alt.Chart(melted).mark_line().encode(
                x='Date', y='Contracts', color=alt.Color('Type', scale=alt.Scale(range=['green', 'red']))
            ).properties(height=250)
            st.altair_chart(chart_ls, use_container_width=True)
            
            st.markdown("##### Flussi Settimanali")
            raw = cot_base['raw_data'].head(3).copy()
            raw["Δ Long"] = raw["Long"] - raw["Long"].shift(-1)
            raw["Δ Short"] = raw["Short"] - raw["Short"].shift(-1)
            raw["Δ Net"] = raw["Net"] - raw["Net"].shift(-1)
            st.dataframe(raw.head(2)[["Date", "Long", "Δ Long", "Short", "Δ Short"]], use_container_width=True)
    else:
        st.warning("Dati COT non disponibili (CFTC Error).")

with tab2:
    if seas:
        c_a, c_b = st.columns([1, 2])
        with c_a:
            st.metric("Win Rate Storico", f"{int(seas['win_rate'])}%")
            st.caption("Probabilità mese positivo (10y)")
        with c_b:
            st.markdown("##### Ciclo Annuale Medio (Median Path)")
            chart_df = pd.DataFrame({'Giorno': seas['chart'].index, 'Valore': seas['chart'].values})
            today_line = alt.Chart(pd.DataFrame({'x': [seas['day']]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
            line = alt.Chart(chart_df).mark_line().encode(
                x=alt.X('Giorno', title='Giorno dell\'anno'), 
                y=alt.Y('Valore', scale=alt.Scale(zero=False), title='Indice (Base 100)')
            ).properties(height=250)
            st.altair_chart(line + today_line, use_container_width=True)
            
            # CALCOLO PENDENZA CICLICA
            curr_day = seas['day']
            trend_30 = "ND"
            try:
                today_val = seas['chart'].iloc[curr_day - 1]
                future_idx = min(curr_day + 30, 364)
                future_val = seas['chart'].iloc[future_idx]
                
                if future_val > today_val * 1.005: 
                    trend_30 = "RIALZISTA 📈" 
                    st.metric("Ciclo (30gg)", trend_30, delta="Positivo", delta_color="normal")
                elif future_val < today_val * 0.995: 
                    trend_30 = "RIBASSISTA 📉"
                    st.metric("Ciclo (30gg)", trend_30, delta="Negativo", delta_color="inverse")
                else: 
                    trend_30 = "LATERALE ➡️"
                    st.metric("Ciclo (30gg)", trend_30, delta="Neutro", delta_color="off")
            except: pass

with tab3:
    if sent["status"] == "OK":
        c_a, c_b = st.columns(2)
        c_a.metric("Retail Short 🔴", f"{sent['short']}%")
        c_b.metric("Retail Long 🟢", f"{sent['long']}%")
        st.progress(sent['long']/100)
        
        st.info(f"📊 Volume Totale: {sent['vol']}")
        
        if sent['long'] > 60:
            st.success("✅ SEGNALE: La folla sta comprando -> Cerca SHORT")
        elif sent['short'] > 60:
            st.success("✅ SEGNALE: La folla sta vendendo -> Cerca LONG")
        else:
            st.warning("⚠️ SEGNALE: Sentiment equilibrato")
    else:
        st.error(f"Dati Sentiment non disponibili: {sent.get('msg', 'Unknown')}")

st.markdown("---")

# --- CHAT AI ---
st.subheader("💬 Chatta con l'Analista AI")

if gemini_key and selected_model:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    prompt_btn = None
    if c_p1.button("📊 Report Completo"): prompt_btn = f"Fammi un report completo su {sel_asset} analizzando COT, Sentiment e Tecnica."
    if c_p2.button("⚖️ Analisi COT"): prompt_btn = "Analizza nel dettaglio il posizionamento istituzionale (COT) e lo Z-Score."
    if c_p3.button("🚦 Consiglio Operativo"): prompt_btn = "Dammi un consiglio operativo (Buy/Sell/Wait) basato sulla confluenza dei dati."
    if c_p4.button("🧠 Spiegami i Rischi"): prompt_btn = "Quali sono i principali rischi tecnici o macro per un trade adesso?"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Chiedi qualcosa...")
    if prompt_btn: user_input = prompt_btn 

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        seas_trend_txt = trend_30 if 'trend_30' in locals() else "ND"
        
        context_str = f"""
        ASSET: {sel_asset}
        DATI LIVE:
        - Prezzo: {tech['price']} (Trend: {tech['trend_desc']})
        - RSI (Wilder): {tech['rsi']:.1f}
        - Forza Relativa: {pwr['base']:.1f}/10 vs USD {pwr['usd']:.1f}/10
        - COT Z-Score: {cot_base['z_score'] if cot_base else 'ND'} (USD Z: {cot_usd['z_score'] if cot_usd else 'ND'})
        - COT Percentile Rank: {cot_base['percentile'] if cot_base else 'ND'}%
        - Sentiment Retail: {sent['long'] if sent['status']=='OK' else 0}% Long
        - Stagionalità Win Rate: {int(seas['win_rate']) if seas else 0}%
        - Ciclo Stagionale Prossimi 30gg: {seas_trend_txt}
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Ragionando..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel(selected_model)
                    full_prompt = f"Sei un Senior Trader. Rispondi basandoti su questi dati:\n{context_str}\n\nDOMANDA: {user_input}"
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Errore: {e}")
else:
    st.info("Inserisci la chiave API per attivare la chat.")