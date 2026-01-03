import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from bs4 import BeautifulSoup
import io
import google.generativeai as genai
from datetime import datetime
from config import SENTIMENT_URL

# ==============================================================================
# MOTORE AI
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
# MOTORE 1: TECNICA (RSI Wilder, MACD, ATR)
# ==============================================================================
@st.cache_data(ttl=300) 
def get_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        curr = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg = ((curr - prev) / prev) * 100
        sma200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        
        # 1. RSI (Wilder's Smoothing)
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 2. MACD (12, 26, 9)
        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # 3. ATR
        high_low = hist["High"] - hist["Low"]
        h_c = np.abs(hist["High"] - hist["Close"].shift())
        l_c = np.abs(hist["Low"] - hist["Close"].shift())
        tr = np.max(pd.concat([high_low, h_c, l_c], axis=1), axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        status = "NEUTRO"
        if sma200:
            status = "ALZISTA" if curr > sma200 else "RIBASSISTA"
            
        return {
            "price": curr, "change": chg, "trend_desc": status, 
            "rsi": rsi, "atr": atr, "sma200": sma200,
            "macd": macd_line.iloc[-1], "macd_sig": signal_line.iloc[-1], "macd_hist": macd_hist
        }
    except: return None

# ==============================================================================
# MOTORE 5: CONFLUENCE SCORE (USER CUSTOM LOGIC)
# ==============================================================================
def calculate_confluence_score(tech, cot_base, cot_usd, sent, seas, pwr):
    score = 0
    reasons = []
    
    # 1. COT (+/- 3) - MAIN DRIVER
    # Logic: Asset Strong (Z>1) AND USD Weak (Z<-1) -> +3
    cot_score = 0
    if cot_base and cot_usd:
        base_z = cot_base['z_score']
        usd_z = cot_usd['z_score']
        
        if base_z > 0.5 and usd_z < -0.5:
            cot_score = 3
            reasons.append("COT: Funds Long Base & Short USD (+3)")
        elif base_z < -0.5 and usd_z > 0.5:
            cot_score = -3
            reasons.append("COT: Funds Short Base & Long USD (-3)")
        elif base_z > 1.0:
            cot_score = 1.5
            reasons.append("COT: Funds Long Base (No USD Div) (+1.5)")
        elif base_z < -1.0:
            cot_score = -1.5
            reasons.append("COT: Funds Short Base (No USD Div) (-1.5)")
            
        # 1.1 COT MOMENTUM (User Request: "Aumento contratti Long")
        # Check if Institutions are ADDING to positions vs previous week
        try:
            hist = cot_base['history_full']
            if len(hist) >= 2:
                # Note: history_full is indexed by Date, sorted descending in analyze_cot_pro? 
                # Let's check analyze_cot_pro implementation. 
                # It does: lookback.set_index("Date"). 
                # lookback is head(period) of d. d is sorted Date DESC.
                # So iloc[0] is Current, iloc[1] is Previous.
                curr_l = hist.iloc[0]['Long']
                prev_l = hist.iloc[1]['Long']
                curr_s = hist.iloc[0]['Short']
                prev_s = hist.iloc[1]['Short']
                
                # Calculate Net Flow: (Change in Longs) - (Change in Shorts)
                delta_long = curr_l - prev_l
                delta_short = curr_s - prev_s
                net_flow = delta_long - delta_short
                
                if net_flow > 0:
                    cot_score += 0.5
                    reasons.append("COT: Net Institutional Buying (+0.5)")
                elif net_flow < 0:
                    cot_score -= 0.5
                    reasons.append("COT: Net Institutional Selling (-0.5)")
        except: pass
            
    score += cot_score
    
    # 2. SENTIMENT (+/- 2)
    if sent and sent['status'] == "OK":
        if sent['long'] > 60: 
            score -= 2
            reasons.append(f"Sentiment: Retail Long {sent['long']}% -> Bearish (-2)")
        elif sent['short'] > 60: 
            score += 2
            reasons.append(f"Sentiment: Retail Short {sent['short']}% -> Bullish (+2)")
            
    # 3. SEASONALITY (+/- 2)
    if seas:
        if seas['win_rate'] > 60: 
            score += 2
            reasons.append(f"Seasonality: Win Rate {int(seas['win_rate'])}% > 60% (+2)")
        elif seas['win_rate'] < 40: 
            score -= 2
            reasons.append(f"Seasonality: Win Rate {int(seas['win_rate'])}% < 40% (-2)")
            
    # 4. RELATIVE STRENGTH (+/- 2)
    if pwr:
        if pwr['base'] > pwr['usd'] + 1:
            score += 2
            reasons.append("Rel Strength: Base > USD (+2)")
        elif pwr['usd'] > pwr['base'] + 1:
            score -= 2
            reasons.append("Rel Strength: USD > Base (-2)")

    # 5. TECHNICAL TREND (+/- 1)
    if tech:
        if tech['trend_desc'] == "ALZISTA": 
            score += 1
            reasons.append("Trend: SMA200 Bullish (+1)")
        elif tech['trend_desc'] == "RIBASSISTA": 
            score -= 1
            reasons.append("Trend: SMA200 Bearish (-1)")

    # LABEL
    if score >= 5: label = "STRONG BUY"
    elif score >= 2: label = "BUY"
    elif score <= -5: label = "STRONG SELL"
    elif score <= -2: label = "SELL"
    else: label = "NEUTRAL"
    
    return {"score": score, "label": label, "reasons": reasons}
@st.cache_data(ttl=300)
def get_currency_strength(pair_ticker):
    try:
        tickers = f"DX-Y.NYB {pair_ticker}"
        data = yf.download(tickers, period="1mo", progress=False)['Close']
        if data.empty: return None
        
        # Handling different MultiIndex structures from yfinance updates
        if isinstance(data.columns, pd.MultiIndex):
             # Try flatting or accessing directly
            try:
                dxy = data["DX-Y.NYB"]
                pair = data[pair_ticker]
            except KeyError:
                # If structure is different, we might need to be more robust
                return None
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
# MOTORE 2: SENTIMENT
# ==============================================================================
@st.cache_data(ttl=3600)
def get_sentiment_data(target_pair):
    from playwright.sync_api import sync_playwright
    import time
    import asyncio
    import sys
    
    # FIX: Enforce ProactorEventLoop on Windows for Playwright subprocesses
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # User-Agent is CRITICAL for FXSSI to avoid 403 or blocking
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    try:
        with sync_playwright() as p:
            # Add args to be more robust
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
            context = browser.new_context(user_agent=UA)
            page = context.new_page()
            
            url = f"{SENTIMENT_URL}?filter={target_pair}"
            print(f"DEBUG: Fetching {url} with Playwright...")
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            # Selector derived from inspection: .cur-rat-broker.broker-average
            # FIX: The page loads ALL pairs. We must find the container for OUR target pair.
            # 1. Find the container that has the text of the pair (e.g. "EURUSD")
            container = page.locator(".cur-rat-cur-pair").filter(has_text=target_pair).first
            
            # 2. Inside that container, find the broker-average row
            selector = ".cur-rat-broker.broker-average"
            
            try:
                # Wait for the container to be visible/attached
                container.wait_for(state="attached", timeout=10000)
                
                # Get the element handle within the container
                row = container.locator(selector).first
                
                # Extract percentages
                left_el = row.locator(".ratio-bar-left")
                right_el = row.locator(".ratio-bar-right")
                
                l_txt = left_el.inner_text(timeout=1000).replace("%", "").strip()
                r_txt = right_el.inner_text(timeout=1000).replace("%", "").strip()
                
                if l_txt and r_txt:
                    long_pct = float(l_txt)
                    short_pct = float(r_txt)
                    browser.close()
                    return {"status": "OK", "long": int(long_pct), "short": int(short_pct), "vol": "N/A"}
                    
            except Exception as e:
                browser.close()
                return {"status": "Err", "msg": f"Missing Data for {target_pair}: {str(e)[:50]}"}
            
            browser.close()
            return {"status": "Err", "msg": "Data missing"}
            
    except Exception as e: return {"status": "Err", "msg": str(e)}

# ==============================================================================
# MOTORE 3: COT (Percentile Rank & Z-Score)
# ==============================================================================
@st.cache_data(ttl=86400)
def download_cot_data():
    from config import CFTC_URL_TEMPLATE
    
    dfs = []
    current_year = datetime.now().year
    # Scarica quest'anno e i 3 precedenti per avere storico sufficiente
    years = range(current_year, current_year - 4, -1)
    
    success = False
    
    for year in years:
        url = CFTC_URL_TEMPLATE.format(year=year)
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
                dfs.append(df)
                success = True
        except Exception as e:
            print(f"Warning: Could not download COT data for {year}: {e}")
            continue
            
    if success and dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Rimuovi duplicati se ci sono sovrapposizioni o errori
        full_df = full_df.drop_duplicates(subset=['Asset', 'Date'])
        return full_df, True
        
    return pd.DataFrame(), False

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
    
    lookback_period = 156 if len(d) >= 156 else len(d)
    lookback = d.head(lookback_period).copy()
    curr = lookback.iloc[0]["Net"]
    
    std = lookback["Net"].std()
    mean = lookback["Net"].mean()
    z_score = (curr - mean) / std if std != 0 else 0
    percentile = (lookback["Net"] < curr).mean() * 100
    
    return {
        "name": best, "net": curr, "z_score": z_score, "percentile": percentile,
        # Returning only necessary data series for charts
        "history": lookback.set_index("Date")["Net"], 
        "history_full": lookback.set_index("Date")[["Net", "Long", "Short", "OpenInt"]],
        "raw_data": d
    }

# ==============================================================================
# MOTORE 4: STAGIONALITÀ
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
        
        avg_path = piv_norm.median(axis=1) # MEDIANA
        avg_path_smooth = avg_path.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        return {"win_rate": win_rate, "chart": avg_path_smooth, "day": datetime.now().timetuple().tm_yday}
    except: return None

# ==============================================================================
# MOTORE 6: MARKET SCANNER
# ==============================================================================
def get_market_snapshot(watchlist, progress_bar=None):
    """
    Fetches raw data for watchlist. 
    COT data is added later in main.py because it requires the large DF.
    """
    results = []
    total = len(watchlist)
    
    for i, ticker in enumerate(watchlist):
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"Scanning {ticker}...")
            
        t_sym = ticker + "=X" if len(ticker) == 6 else ticker # Simple heuristics
        
        # 1. Tech
        tech = get_technical_analysis(t_sym)
        # 2. Power
        pwr = get_currency_strength(t_sym)
        # 3. Sentiment (Cached)
        sent = get_sentiment_data(ticker)
        # 4. Seasonality
        seas = get_seasonality_pro(t_sym)
        
        results.append({
            "Ticker": ticker,
            "Tech": tech,
            "Pwr": pwr,
            "Sent": sent,
            "Seas": seas
        })
        
    return results

def finalize_scanner_scores(snapshot_data, cot_df):
    """
    Combines snapshot data with COT data to produce final scores.
    """
    final_rows = []
    cot_usd = analyze_cot_pro(cot_df, "USD") # Calculate USD once
    
    for item in snapshot_data:
        ticker = item['Ticker']
        base = ticker[:3] # Heuristic: First 3 chars
        
        cot_base = analyze_cot_pro(cot_df, base)
        
        # Calculate Score
        conf = calculate_confluence_score(
            item['Tech'], cot_base, cot_usd, item['Sent'], item['Seas'], item['Pwr']
        )
        
        # Create summary row
        final_rows.append({
            "Asset": ticker,
            "Score": conf['score'],
            "Bias": conf['label'],
            "Signal": conf['reasons'] # For tooltip or icons
        })
        
    return pd.DataFrame(final_rows)

# ==============================================================================
# MOTORE 6: MARKET SCANNER (BATCH PROCESSING)
# ==============================================================================
def get_market_snapshot(watchlist, progress_bar=None):
    """
    Calculates Confluence Score for all assets in watchlist.
    Returns a list of dicts.
    """
    results = []
    total = len(watchlist)
    
    for i, ticker in enumerate(watchlist):
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"Analyzing {ticker}...")
            
        # 1. Fetch Data
        tech = get_technical_analysis(ticker + "=X" if "USD" in ticker and "X" not in ticker else ticker)
        
        # Determine Base/Quote for COT/Strength
        # Simplying for majors: Base is first 3 chars, Quote is last 3 (usually USD)
        is_forex = len(ticker) == 6
        base = ticker[:3] if is_forex else ticker
        quote = "USD" # Default assumption for now
        
        # COT
        # We need to access the cached COT dataframe from session state or reload it
        # Since this function is in data.py, we might not have direct access to st.session_state's COT data perfectly
        # But we can try to call analyze_cot_pro if we pass the dataframe. 
        # For simplicity in this iteration, we'll skip COT in the scanner if we can't easily pass the DF,
        # OR better: We rely on the caller to pass the COT dataframe.
        # Let's assume we can't easily get COT here without the DF. 
        # IMPROVEMENT: We will fetch Tech, Sentiment, Seasonality here.
        
        # Currency Strength
        pwr = get_currency_strength(ticker + "=X" if is_forex else ticker)
        
        # Sentiment (This is the slow part)
        # We might skip live sentiment for the scanner to be fast, or use cached.
        # Let's try to get it.
        sent = get_sentiment_data(ticker)
        
        # Seasonality
        seas = get_seasonality_pro(ticker + "=X" if is_forex else ticker)
        
        # Calculate Score Partial (We might miss COT here if not passed)
        # To make this robust, we should refactor to allow passing COT data or fetching it here.
        # For V1, let's call calculate_confluence_score with None for COT if we don't have it.
        # But COT is the main driver! 
        # Let's fetch COT data locally if possible, or skip it.
        # Actually, main.py has the COT dataframe. We should pass it.
        # See below for signature update.
        
        results.append({
            "Ticker": ticker,
            "Tech": tech,
            "Pwr": pwr,
            "Sent": sent,
            "Seas": seas
        })
        
    return results

def calculate_score_for_snapshot(snapshot_item, cot_df):
    """
    Helper to calculate score when we have the COT DF available.
    """
    ticker = snapshot_item['Ticker']
    tech = snapshot_item['Tech']
    pwr = snapshot_item['Pwr']
    sent = snapshot_item['Sent']
    seas = snapshot_item['Seas']
    
    base = ticker[:3]
    # Analyze COT
    cot_base = analyze_cot_pro(cot_df, base)
    cot_usd = analyze_cot_pro(cot_df, "USD")
    
    confluence = calculate_confluence_score(tech, cot_base, cot_usd, sent, seas, pwr)
    return confluence
