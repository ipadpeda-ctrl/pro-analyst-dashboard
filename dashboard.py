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

st.set_page_config(page_title="Pro Trend Dashboard", layout="wide")

# ==============================================================================
# CSS PERSONALIZZATO
# ==============================================================================
st.markdown("""
<style>
div.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.st-emotion-cache-vj0p0d.ezrtsby2 {text-align: center;}
div[data-testid="stMetricValue"] {font-size: 1.6rem !important;}
[data-testid="stProgress"] {margin-top: -10px !important; margin-bottom: 10px !important;}
</style>
""", unsafe_allow_html=True)

# 🔗 CONFIGURAZIONE
current_year = datetime.now().year
CFTC_URL = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
SENTIMENT_URL = "https://www.myfxbook.com/community/outlook"

# --- MOTORE AI: LISTA MODELLI DINAMICA ---
def get_available_models(api_key):
    """Chiede a Google quali modelli sono disponibili per questa chiave."""
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except:
        return []

def get_ai_analysis(api_key, model_name, context_data):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Agisci come un Senior Hedge Fund Trader. Analizza questi dati di mercato per {context_data['asset']}:
        
        1. MACRO COT: Z-Score Asset: {context_data['cot_z']:.2f} | Z-Score USD: {context_data['usd_z']:.2f}
        2. SENTIMENT RETAIL: {context_data['sent_long']}% Long vs {context_data['sent_short']}% Short.
        3. STAGIONALITÀ: Win Rate {context_data['seas_win']}% | Previsione: {context_data['seas_trend']}
        4. TECNICA: Prezzo {context_data['price']} | Trend Fondo: {context_data['trend']} | RSI: {context_data['rsi']:.1f}
        
        Sintetizza in 3 punti bullet points brevi:
        - Situazione Macro (Confluenza o Conflitto?)
        - Rischi (Timing, livelli tecnici)
        - Verdetto Operativo (Buy, Sell, Wait)
        Sii diretto, professionale e usa emoji.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Errore AI: {str(e)}"

# --- MOTORE 1: ANALISI TECNICA ---
def get_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        curr = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg = ((curr - prev) / prev) * 100
        
        sma200, sma50 = None, None
        status = "Dati Insufficienti ❓"
        
        if len(hist) >= 200:
            sma50 = hist["Close"].rolling(50).mean().iloc[-1]
            sma200 = hist["Close"].rolling(200).mean().iloc[-1]
            if curr > sma200:
                status = "ALZISTA (Sopra SMA200) 🟢"
                if curr < sma50: status += " [In Ritracciamento ⚠️]"
            else:
                status = "RIBASSISTA (Sotto SMA200) 🔴"
                if curr > sma50: status += " [Possibile Inversione ⚠️]"
        
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
            
        return {"price": curr, "change": chg, "trend_desc": status, "rsi": rsi, "atr": atr, "sma200": sma200}
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
    mn, mx = lookback["Net"].min(), lookback["Net"].max()
    std = lookback["Net"].std()
    mean = lookback["Net"].mean()
    idx = ((curr - mn) / (mx - mn)) * 100 if mx != mn else 50
    z_score = (curr - mean) / std if std != 0 else 0
    
    return {
        "name": best, "net": curr, "index": idx, 
        "z_score": z_score, 
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
        
        m_ret = hist['Close'].resample('ME').ffill().pct_change()
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
# DASHBOARD
# ==============================================================================
st.title("🦅 Pro Analyst Dashboard")

# --- GUIDA ---
with st.expander("📘 GUIDA RAPIDA"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**COT:** Z-Score > 2 (Long Estremo), < -2 (Short Estremo).")
        st.markdown("**Open Interest:** Se Prezzo sale + OI sale = Trend Sano.")
    with c2:
        st.markdown("**Sentiment:** Contrarian (Fai l'opposto della massa).")

df_cot, cot_ok = download_cot_data(CFTC_URL)

# SIDEBAR
st.sidebar.header("⚙️ Asset Manager")
asset_map = {
    "EUR/USD":   {"base": "EURO FX", "usd": "USD INDEX", "yf": "EURUSD=X", "myfx": "EURUSD"},
    "GBP/USD":   {"base": "BRITISH POUND", "usd": "USD INDEX", "yf": "GBPUSD=X", "myfx": "GBPUSD"},
    "Gold":      {"base": "GOLD", "usd": "USD INDEX", "yf": "GC=F", "myfx": "XAUUSD"}, 
    "Crude Oil": {"base": "CRUDE OIL", "usd": "USD INDEX", "yf": "CL=F", "myfx": "WTI"}, 
    "Bitcoin":   {"base": "BITCOIN", "usd": "USD INDEX", "yf": "BTC-USD", "myfx": "BTCUSD"},
    "S&P 500":   {"base": "E-MINI S&P 500", "usd": "USD INDEX", "yf": "ES=F", "myfx": "SP500"}
}
sel_asset = st.sidebar.selectbox("Seleziona Asset", list(asset_map.keys()))
cfg = asset_map[sel_asset]

# --- AI KEY INPUT & SELECTOR ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Copilot")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

# Selettore Modello Dinamico
selected_model = None
if gemini_key:
    available_models = get_available_models(gemini_key)
    if available_models:
        st.sidebar.success(f"✅ Trovati {len(available_models)} modelli")
        selected_model = st.sidebar.selectbox("Scegli Modello", available_models, index=0)
    else:
        st.sidebar.warning("Chiave valida ma nessun modello 'generateContent' trovato.")

# HEADER
tech = get_technical_analysis(cfg["yf"])
if tech:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prezzo", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
    trend_txt = tech["trend_desc"].split(" ")[0] if tech["sma200"] is not None else "N/A"
    c2.metric("Trend di Fondo", trend_txt) 
    rsi_c = "inverse" if tech['rsi'] > 70 else "normal"
    c3.metric("RSI (14)", f"{tech['rsi']:.1f}", "Ipercomprato" if tech['rsi']>70 else "Neutro", delta_color=rsi_c)
    c4.metric("Volatilità (ATR)", f"{tech['atr']:.4f}")
    
st.markdown("---")

col1, col2, col3 = st.columns(3)

# VARIABILI DATI PER AI
ai_context = {}

# 1. MACRO COT
with col1:
    st.header("1. Macro COT ⚖️")
    if cot_ok:
        base = analyze_cot_pro(df_cot, cfg["base"])
        usd = analyze_cot_pro(df_cot, cfg["usd"])
        
        if base and usd:
            st.write(f"**{sel_asset.split('/')[0]}**: Z-Score {base['z_score']:.2f}")
            st.area_chart(base['history'], height=80, color="#29b5e8")
            st.write(f"**USD**: Z-Score {usd['z_score']:.2f}")
            st.area_chart(usd['history'], height=80, color="#ffaa00")
            
            # Dati per AI
            ai_context['cot_z'] = base['z_score']
            ai_context['usd_z'] = usd['z_score']
            
            with st.expander(f"🔬 Struttura: {sel_asset}"):
                st.markdown(f"##### {sel_asset}: Long vs Short")
                hist_data = base['history_full'].reset_index()
                melted = hist_data.melt('Date', value_vars=['Long', 'Short'], var_name='Type', value_name='Contracts')
                chart_ls = alt.Chart(melted).mark_line().encode(
                    x='Date', y='Contracts',
                    color=alt.Color('Type', scale=alt.Scale(domain=['Long', 'Short'], range=['green', 'red'])),
                    tooltip=['Date', 'Type', 'Contracts']
                ).properties(height=200)
                st.altair_chart(chart_ls, use_container_width=True)
                
                st.markdown(f"##### Open Interest")
                st.bar_chart(base['history_full']['OpenInt'], height=150)
                
                st.markdown("##### Flussi Settimanali")
                raw = base['raw_data'].head(3).copy()
                raw["Δ Long"] = raw["Long"] - raw["Long"].shift(-1)
                raw["Δ Short"] = raw["Short"] - raw["Short"].shift(-1)
                raw["Δ Net"] = raw["Net"] - raw["Net"].shift(-1)
                display_table = raw.head(2)[["Date", "Long", "Δ Long", "Short", "Δ Short", "Net"]].copy()
                display_table["Date"] = display_table["Date"].dt.strftime('%d/%m')
                st.dataframe(display_table.style.format({"Long": "{:,.0f}", "Short": "{:,.0f}", "Net": "{:,.0f}", "Δ Long": "{:+,.0f}", "Δ Short": "{:+,.0f}"}), hide_index=True, use_container_width=True)

            score_cot = 0
            if base['z_score'] > 0.5 and usd['z_score'] < -0.5:
                st.success("✅ STRONG LONG")
                score_cot = 1
            elif base['z_score'] < -0.5 and usd['z_score'] > 0.5:
                st.error("🔻 STRONG SHORT")
                score_cot = -1
            else: st.info("⚖️ Neutrale")
        else: st.warning("Dati COT insufficienti")
    else: st.error("Err COT")

# 2. STAGIONALITÀ
with col2:
    st.header("2. Stagionalità 📅")
    seas = get_seasonality_pro(cfg["yf"])
    if seas:
        c_win, c_avg = st.columns(2)
        c_win.metric("Win Rate (10y)", f"{int(seas['win_rate'])}%")
        c_avg.metric("Ritorno", f"{seas['avg_return']:.2f}%")
        
        st.markdown("**Ciclo Annuale (Smoothed)**")
        chart_df = pd.DataFrame({'Giorno': seas['chart'].index, 'Valore': seas['chart'].values})
        today_line = alt.Chart(pd.DataFrame({'x': [seas['day']]})).mark_rule(color='red').encode(x='x')
        line = alt.Chart(chart_df).mark_line().encode(
            x=alt.X('Giorno', title='Giorno'), y=alt.Y('Valore', scale=alt.Scale(zero=False), title='Idx'), tooltip=['Giorno', 'Valore']
        ).properties(height=200)
        st.altair_chart(line + today_line, use_container_width=True)
        
        curr_day = seas['day']
        try:
            today_val = seas['chart'].iloc[curr_day - 1]
            future_idx = min(curr_day + 30, 364)
            future_val = seas['chart'].iloc[future_idx]
            trend = "RIALZISTA 📈" if future_val > today_val * 1.005 else "RIBASSISTA 📉" if future_val < today_val * 0.995 else "LATERALE ➡️"
            st.info(f"🔮 Previsione 30gg: **{trend}**")
            
            # Dati AI
            ai_context['seas_win'] = int(seas['win_rate'])
            ai_context['seas_trend'] = trend
        except: pass
            
        score_seas = 0
        if seas['win_rate'] >= 65: 
            st.success("✅ Mese Forte")
            score_seas = 1
        elif seas['win_rate'] <= 35: 
            st.error("🔻 Mese Debole")
            score_seas = -1
        else: st.warning("⚠️ Mese Misto")
    else: st.write("Loading...")

# 3. SENTIMENT
with col3:
    st.header("3. Sentiment 🐑")
    sent = get_sentiment_data(cfg["myfx"])
    if sent["status"] == "OK":
        c_s, c_l = st.columns(2)
        c_s.metric("Short", f"{sent['short']}%", delta_color="inverse")
        c_l.metric("Long", f"{sent['long']}%")
        st.progress(sent['long']/100)
        st.caption(f"Vol: {sent['vol']}")
        
        # Dati AI
        ai_context['sent_long'] = sent['long']
        ai_context['sent_short'] = sent['short']
        
        score_sent = 0
        if sent['long'] > 60: 
            st.success("✅ Crowd Long -> SHORT")
            score_sent = -1
        elif sent['short'] > 60: 
            st.success("✅ Crowd Short -> LONG")
            score_sent = 1
        else: st.warning("⚠️ Neutrale")
    else: st.error("No Data")

st.markdown("---")

# SCORE & AI ANALYST
if 'score_cot' in locals() and 'score_seas' in locals() and 'score_sent' in locals():
    total = score_cot + score_seas + score_sent
    st.subheader("🎯 VERDETTO FINALE")
    
    # Prepariamo dati AI finali
    if tech:
        ai_context['asset'] = sel_asset
        ai_context['price'] = f"{tech['price']:.4f}"
        ai_context['trend'] = tech['trend_desc']
        ai_context['rsi'] = tech['rsi']
    
    c_fin, c_chk = st.columns([2,1])
    with c_fin:
        if total >= 2: st.success(f"🔥 BUY OPPORTUNITY (Score {total}/3)")
        elif total <= -2: st.error(f"❄️ SELL OPPORTUNITY (Score {total}/3)")
        else: st.info(f"✋ WAIT / HOLD (Score {total}/3)")
        
        # --- PULSANTE AI ---
        if gemini_key and selected_model:
            if st.button("🤖 CHIEDI ALL'ANALISTA AI"):
                with st.spinner(f"Analizzando con {selected_model}..."):
                    analysis = get_ai_analysis(gemini_key, selected_model, ai_context)
                    st.markdown("### 🧠 Analisi AI")
                    st.markdown(analysis)
        elif total != 0:
            st.caption("Inserisci la Gemini API Key e scegli un modello.")

    with c_chk:
        if tech and tech['sma200']:
            if tech['rsi']>70: st.warning("⚠️ RSI Ipercomprato")
            elif tech['rsi']<30: st.warning("⚠️ RSI Ipervenduto")
            elif total>=2 and tech['price']<tech['sma200']: st.warning("⚠️ Sotto SMA200")
            elif total<=-2 and tech['price']>tech['sma200']: st.warning("⚠️ Sopra SMA200")
            else: st.success("✅ Timing Tecnico OK")