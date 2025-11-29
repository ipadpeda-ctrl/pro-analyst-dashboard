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
# CSS PERSONALIZZATO
# ==============================================================================
st.markdown("""
<style>
div.block-container {padding-top: 1rem; padding-bottom: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.4rem !important;}
div[data-testid="stMetricLabel"] {font-size: 0.9rem !important; font-weight: bold; color: #888;}
button[data-baseweb="tab"] {font-size: 1.1rem; font-weight: 600;}
.signal-box {padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;}
.bullish {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
.bearish {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
.neutral {background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;}
</style>
""", unsafe_allow_html=True)

# 🔗 CONFIGURAZIONE
current_year = datetime.now().year
CFTC_URL = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
SENTIMENT_URL = "https://www.myfxbook.com/community/outlook"

# --- MOTORE AI (VERSIONE UTENTE CON SELETTORE) ---
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
        Analisi Flash per Trader ({context_data['asset']}):
        1. COT: Z-Score Asset {context_data['cot_z']:.2f} | USD {context_data['usd_z']:.2f}
        2. SENTIMENT: {context_data['sent_long']}% Long / {context_data['sent_short']}% Short
        3. STAGIONALITÀ: Win {context_data['seas_win']}% | Trend: {context_data['seas_trend']}
        4. TECNICA: Prezzo {context_data['price']} | Trend: {context_data['trend']} | RSI: {context_data['rsi']:.1f}
        
        Dammi SOLO:
        1. Il Verdetto (Bullish/Bearish/Neutral)
        2. Il motivo principale (es. "Confluenza COT e Sentiment")
        3. Un livello di attenzione (es. "Attenti all'RSI alto")
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "🚦 Limite richieste raggiunto. Attendi 1 min."
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
        
        status = "NEUTRO"
        if sma200:
            status = "ALZISTA" if curr > sma200 else "RIBASSISTA"
            
        return {"price": curr, "change": chg, "trend_desc": status, "rsi": rsi, "atr": atr, "sma200": sma200}
    except: return None

# --- MOTORE 1.5: FORZA RELATIVA (POWER METER) ---
def get_currency_strength(pair_ticker):
    try:
        # Scarica DXY (Dollaro) e la Coppia
        tickers = f"DX-Y.NYB {pair_ticker}"
        data = yf.download(tickers, period="5d", progress=False)['Close']
        
        if data.empty: return None
        
        # Gestione colonne
        if isinstance(data.columns, pd.MultiIndex):
            dxy = data["DX-Y.NYB"]
            pair = data[pair_ticker]
        else:
            dxy = data["DX-Y.NYB"]
            pair = data[pair_ticker]

        # Calcolo % Change 24h
        dxy_chg = ((dxy.iloc[-1] - dxy.iloc[-2]) / dxy.iloc[-2]) * 100
        pair_chg = ((pair.iloc[-1] - pair.iloc[-2]) / pair.iloc[-2]) * 100
        
        # Logica Approssimata
        usd_score = 5 + (dxy_chg * 4) 
        base_chg_est = pair_chg + dxy_chg
        base_score = 5 + (base_chg_est * 4)
        
        # Clip tra 0 e 10
        usd_score = max(0, min(10, usd_score))
        base_score = max(0, min(10, base_score))
        
        return {"base": base_score, "usd": usd_score}
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
# DASHBOARD START
# ==============================================================================
st.title("🦅 Pro Analyst Dashboard")

df_cot, cot_ok = download_cot_data(CFTC_URL)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configurazione")
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
    # GESTIONE CHIAVE UTENTE
    if "GEMINI_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_KEY"]
        st.success("🔑 AI Ready")
    else:
        gemini_key = st.text_input("API Key (Opzionale)", type="password")
    
    # SELETTORE MODELLO (RIPRISTINATO)
    selected_model = None
    if gemini_key:
        mods = get_filtered_models(gemini_key)
        if mods: selected_model = st.selectbox("Modello", mods)

# --- CALCOLI ---
tech = get_technical_analysis(cfg["yf"])
pwr = get_currency_strength(cfg["yf"]) # NUOVO POWER METER
seas = get_seasonality_pro(cfg["yf"])
sent = get_sentiment_data(cfg["myfx"])
cot_base, cot_usd = None, None
if cot_ok:
    cot_base = analyze_cot_pro(df_cot, cfg["base"])
    cot_usd = analyze_cot_pro(df_cot, cfg["usd"])

# --- TOP BAR ---
total_score = 0
cot_score = 0
if cot_base and cot_usd:
    if cot_base['z_score'] > 0.5 and cot_usd['z_score'] < -0.5: cot_score = 1
    elif cot_base['z_score'] < -0.5 and cot_usd['z_score'] > 0.5: cot_score = -1
seas_score = 0
if seas and seas['win_rate'] >= 65: seas_score = 1
elif seas and seas['win_rate'] <= 35: seas_score = -1
sent_score = 0
if sent["status"] == "OK":
    if sent['long'] > 60: sent_score = -1
    elif sent['short'] > 60: sent_score = 1
total_score = cot_score + seas_score + sent_score

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
if tech:
    c1.metric("Prezzo", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
    c2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta_color="off")
    c3.metric("Trend", tech['trend_desc'], delta_color="off")

with c4:
    if total_score >= 2:
        st.markdown('<div class="signal-box bullish">🔥 STRONG BUY OPPORTUNITY</div>', unsafe_allow_html=True)
    elif total_score <= -2:
        st.markdown('<div class="signal-box bearish">❄️ STRONG SELL OPPORTUNITY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-box neutral">✋ MARKET NEUTRAL / WAIT</div>', unsafe_allow_html=True)

# --- SEZIONE FORZA RELATIVA (NUOVA) ---
if pwr:
    st.markdown("### 💪 Forza Relativa (Power Meter)")
    cp1, cp2 = st.columns(2)
    asset_name = sel_asset.split('/')[0]
    with cp1:
        st.write(f"**{asset_name}**: {pwr['base']:.1f}/10")
        st.progress(pwr['base'] / 10)
    with cp2:
        st.write(f"**USD**: {pwr['usd']:.1f}/10")
        st.progress(pwr['usd'] / 10)
    st.markdown("---")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Analisi Macro (COT)", "📅 Stagionalità & Cicli", "🐑 Sentiment & Volumi"])

with tab1:
    if cot_base and cot_usd:
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"#### {sel_asset.split('/')[0]}")
            st.metric("Z-Score (Forza)", f"{cot_base['z_score']:.2f}", help="Sopra 2.0 è ipercomprato")
            st.area_chart(cot_base['history'], height=150, color="#29b5e8")
        with c_b:
            st.markdown("#### USD (Dollaro)")
            st.metric("Z-Score (Forza)", f"{cot_usd['z_score']:.2f}")
            st.area_chart(cot_usd['history'], height=150, color="#ffaa00")
            
        with st.expander("🔬 Analisi Approfondita (Long/Short & OI)"):
            st.markdown(f"##### {sel_asset}: Long vs Short")
            hist_data = cot_base['history_full'].reset_index()
            melted = hist_data.melt('Date', value_vars=['Long', 'Short'], var_name='Type', value_name='Contracts')
            chart_ls = alt.Chart(melted).mark_line().encode(
                x='Date', y='Contracts', color=alt.Color('Type', scale=alt.Scale(range=['green', 'red']))
            ).properties(height=250)
            st.altair_chart(chart_ls, use_container_width=True)
            
            st.markdown(f"##### Open Interest")
            st.bar_chart(cot_base['history_full']['OpenInt'], height=150)
            
            st.markdown("##### Flussi Settimanali")
            raw = cot_base['raw_data'].head(3).copy()
            raw["Δ Long"] = raw["Long"] - raw["Long"].shift(-1)
            raw["Δ Short"] = raw["Short"] - raw["Short"].shift(-1)
            raw["Δ Net"] = raw["Net"] - raw["Net"].shift(-1)
            display_table = raw.head(2)[["Date", "Long", "Δ Long", "Short", "Δ Short", "Net"]].copy()
            display_table["Date"] = display_table["Date"].dt.strftime('%d/%m')
            st.dataframe(display_table.style.format({"Long": "{:,.0f}", "Short": "{:,.0f}", "Net": "{:,.0f}", "Δ Long": "{:+,.0f}", "Δ Short": "{:+,.0f}"}), hide_index=True, use_container_width=True)
    else:
        st.warning("Dati COT non disponibili.")

with tab2:
    if seas:
        c_a, c_b = st.columns([1, 2])
        with c_a:
            st.metric("Win Rate Storico", f"{int(seas['win_rate'])}%")
            st.caption("Probabilità mese positivo (10y)")
        with c_b:
            st.markdown("##### Ciclo Annuale Medio")
            chart_df = pd.DataFrame({'Giorno': seas['chart'].index, 'Valore': seas['chart'].values})
            today_line = alt.Chart(pd.DataFrame({'x': [seas['day']]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
            line = alt.Chart(chart_df).mark_line().encode(
                x=alt.X('Giorno', title='Giorno dell\'anno'), 
                y=alt.Y('Valore', scale=alt.Scale(zero=False), title='Indice')
            ).properties(height=250)
            st.altair_chart(line + today_line, use_container_width=True)

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
            st.warning("⚠️ SEGNALE: Sentiment equilibrato (Nessun vantaggio)")
    else:
        st.error("Dati Sentiment non disponibili.")

st.markdown("---")

# --- SEZIONE AI (IN BASSO) ---
st.subheader("🧠 Analisi Operativa AI")

if gemini_key and selected_model:
    if st.button("🤖 Chiedi all'Analista"):
        ai_ctx = {
            'asset': sel_asset,
            'cot_z': cot_base['z_score'] if cot_base else 0,
            'usd_z': cot_usd['z_score'] if cot_usd else 0,
            'sent_long': sent['long'] if sent['status']=='OK' else 50,
            'sent_short': sent['short'] if sent['status']=='OK' else 50,
            'seas_win': int(seas['win_rate']) if seas else 50,
            'seas_trend': "ND",
            'price': tech['price'], 'trend': tech['trend_desc'], 'rsi': tech['rsi']
        }
        with st.spinner("Analizzando i mercati..."):
            st.markdown(get_ai_analysis(gemini_key, selected_model, ai_ctx))
else:
    st.info("Per usare l'assistente AI, assicurati che la chiave sia salvata nei Secrets o inseriscila nella Sidebar.")