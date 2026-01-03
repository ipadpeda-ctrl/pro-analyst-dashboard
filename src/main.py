import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import altair as alt
import pandas as pd
import google.generativeai as genai

from config import (
    SENTIMENT_URL, AI_GEMINI_URL, 
    COLOR_PRIMARY, COLOR_BG, COLOR_CARD, COLOR_TEXT, FONT_MAIN,
    ASSET_MAP, WATCHLIST, CUSTOM_CSS
)
from data import (
    get_technical_analysis, get_currency_strength, get_sentiment_data,
    download_cot_data, analyze_cot_pro, get_seasonality_pro, get_filtered_models,
    calculate_confluence_score, get_market_snapshot, finalize_scanner_scores
)
from ui import render_header, render_metric_card

# --- SETUP ---
st.set_page_config(page_title="Pro Trend Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- CLOUD FIX: INSTALL PLAYWRIGHT BROWSER ---
import os
import subprocess
import sys

try:
    # 1. Install Playwright Package if missing (Fallback)
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "playwright", "matplotlib"])
    
    # 2. Install Browsers
    # On Streamlit Cloud this usually runs once
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    print(f"Playwright Install Error: {e}")


# --- GLOBAL STATE ---
if "COT_DATA" not in st.session_state:
    with st.spinner("Downloading Institutional Data..."):
        df, ok = download_cot_data()
        st.session_state["COT_DATA"] = (df, ok)

df_cot, cot_ok = st.session_state["COT_DATA"]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/bullish.png", width=60) # Placeholder Icon
    st.title("Forecaster Pro")
    
    selected = option_menu(
        menu_title=None,
        options=["Market Scanner", "Deep Dive", "AI Analyst", "Settings"],
        icons=["radar", "graph-up", "robot", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#111318"},
            "icon": {"color": "#00e5ff", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"},
        }
    )
    
    st.markdown("---")
    
    # Asset Selection Global - ONLY FOR DEEP DIVE
    sel_asset = "EUR/USD" # Default
    if selected == "Deep Dive":
        sel_asset = st.selectbox("Select Asset", list(ASSET_MAP.keys()))

# --- PAGES ---

if selected == "Market Scanner":
    render_header("Market Scanner", "Global Confluence Matrix")
    
    st.markdown("### 📡 Live Opportunity Scanner")
    st.info("Click below to scan major pairs for Technical, Sentiment, and Institutional confluences.")
    
    if st.button("🚀 START FULL SCAN", type="primary", use_container_width=True):
        with st.spinner("Scanning Markets... (Fetching live data)"):
            # 1. Get Snapshot
            progress = st.progress(0)
            snapshot = get_market_snapshot(WATCHLIST, progress_bar=progress)
            progress.empty()
            
            # 2. Finalize
            if cot_ok:
                df_res = finalize_scanner_scores(snapshot, df_cot)
                
                # RENDER TABLE
                st.markdown("### 🏆 Top Opportunities")
                st.dataframe(
                    df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-10, vmax=10),
                    column_order=["Asset", "Score", "Bias", "Signal"],
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Confluence Score",
                            format="%d",
                            min_value=-10,
                            max_value=10,
                        ),
                        "Bias": st.column_config.TextColumn("Signal Bias"),
                    }
                )
            else:
                st.error("COT Data not loaded. Cannot calculate full scores.")

elif selected == "Deep Dive":
    cfg = ASSET_MAP[sel_asset]
    render_header(f"{sel_asset}", "Deep Dive Analysis")
    
    # FETCH DATA
    tech = get_technical_analysis(cfg["yf"])
    pwr = get_currency_strength(cfg["yf"])
    seas = get_seasonality_pro(cfg["yf"])
    sent = get_sentiment_data(cfg["fxssi"])
    
    cot_base, cot_usd = None, None
    if cot_ok:
        cot_base = analyze_cot_pro(df_cot, cfg["base"])
        cot_usd = analyze_cot_pro(df_cot, cfg["usd"])

    # --- 1. SUMMARY / SIGNAL ---
    confluence = calculate_confluence_score(tech, cot_base, cot_usd, sent, seas, pwr)
    
    st.markdown("### 🔥 Trade Signal")
    sc1, sc2 = st.columns([1, 4])
    with sc1:
        st.metric("SCORE", f"{confluence['score']}/10", confluence['label'])
        
        # AI REPORT BUTTON
        if st.button("📝 AI Report", type="secondary", use_container_width=True):
            gemini_key = st.secrets.get("GEMINI_KEY")
            if not gemini_key:
                st.error("⚠️ Please set GEMINI_KEY in .streamlit/secrets.toml")
            else:
                with st.spinner("🤖 Analyzing Market Structure..."):
                    try:
                        # Prepare Context
                        ctx = f"""
                        ASSET: {sel_asset}
                        SCORE: {confluence['score']}/10 ({confluence['label']})
                        
                        DRIVERS:
                        - Technical: {tech['trend_desc'] if tech else 'N/A'} (RSI: {tech['rsi']:.1f} if tech else 'N/A')
                        - COT (Smart Money): Z-Score {cot_base['z_score']:.2f} (Net Pos) if cot_base else 'N/A'
                        - Sentiment (Retail): {sent['long']}% Long / {sent['short']}% Short
                        - Seasonality: {int(seas['win_rate'])}% Win Rate (Month: {datetime.datetime.now().strftime('%B')})
                        
                        CONFLUENCE FACTORS:
                        {', '.join(confluence['reasons'])}
                        """
                        
                        sys_prompt = """
                        You are a Senior Wall Street Global Macro Strategist.
                        Analyze the provided data and write a concise "Executive Trade Plan" (Max 150 words).
                        Structure:
                        1. **Strategic Bias**: Bullish/Bearish/Neutral with Confidence Level (High/Med/Low).
                        2. **The "Why"**: Synthesize the strongest drivers (e.g. "Smart money is buying into technical uptrend").
                        3. **Key Risks**: What invalidates this view?
                        4. **Execution**: Immediate entry or wait for level?
                        """
                        
                        # Dynamic Model Selection
                        avail_models = get_filtered_models(gemini_key)
                        target_model = avail_models[0] if avail_models else 'gemini-pro'
                        
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel(target_model)
                        resp = model.generate_content(sys_prompt + "\nDATA:\n" + ctx)
                        
                        st.info(f"### 🧠 AI Executive Summary ({target_model})\n\n{resp.text}")
                        
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")

    with sc2:
        st.markdown("**Drivers Analysis:**")
        # VISUAL TRAFFIC LIGHTS
        badges = []
        for r in confluence['reasons']:
            # Parse logic: "COT: Funds Long..." -> Type=COT, Color=Green/Red
            color = "green" if "(+)" in r else "red" if "(-" in r else "gray"
            badge_bg = "#dcfce7" if color == "green" else "#fee2e2" if color == "red" else "#f3f4f6"
            badge_text = "#166534" if color == "green" else "#991b1b" if color == "red" else "#374151"
            
            # Icon Mapping
            icon = "⏺️"
            if "COT" in r: icon = "🏛️"
            elif "Sentiment" in r: icon = "👥"
            elif "Seasonality" in r: icon = "📅"
            elif "Trend" in r: icon = "📈"
            elif "Rel Strength" in r: icon = "💪"
            
            # Clean label
            label = r.split(":")[0] # "COT", "Sentiment"
            
            # Render HTML Badge
            badges.append(f"""
            <span style="
                background-color: {badge_bg};
                color: {badge_text};
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 0.9em;
                font-weight: 600;
                margin-right: 8px;
                display: inline-flex;
                align-items: center;
                border: 1px solid {color};
            ">{icon} {r}</span>
            """)
            
        st.markdown(" ".join(badges), unsafe_allow_html=True)
            
    with st.expander("ℹ️ How is this calculated?"):
        st.markdown("""
        **Score Logic (Max 10):**
        1. **COT (Smart Money)**: ±3 Pts (Divergence between Asset & USD).
        2. **Relative Strength**: ±2 Pts (Asset vs USD strength).
        3. **Sentiment**: ±2 Pts (Contrarian).
        4. **Seasonality**: ±2 Pts (>60% Win Rate).
        5. **Trend (MA200)**: ±1 Pt.
        """)
    
    st.markdown("---")

    # 1. TOP METRICS
    st.markdown("### 📊 Market Data")
    c1, c2, c3, c4, c5 = st.columns(5)
    if tech:
        render_metric_card(c1, "PREZZO", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
        render_metric_card(c2, "RSI (14)", f"{tech['rsi']:.1f}")
        render_metric_card(c3, "MACD Hist", f"{tech['macd_hist']:.4f}", f"Signal: {tech['macd_sig']:.4f}")
        render_metric_card(c4, "TREND", tech['trend_desc'])
        render_metric_card(c5, "ATR", f"{tech['atr']:.4f}")

    st.markdown("### 🧠 Market Intelligence")
    
    # 2. COLUMNS LAYOUT
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # TABS
        tab1, tab2, tab3 = st.tabs(["🏛️ COT Positioning", "📅 Seasonality", "🌊 Sentiment"])
        
        with tab1:
            if cot_base and cot_usd:
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"**{sel_asset.split('/')[0]}** (Net Pos)")
                    st.metric("Z-Score", f"{cot_base['z_score']:.2f}", help="Standard Deviations from Mean")
                    st.area_chart(cot_base['history'], height=150, color="#00e5ff")
                with c_b:
                    st.markdown(f"**USD Index** (Net Pos)")
                    st.metric("Z-Score", f"{cot_usd['z_score']:.2f}")
                    st.area_chart(cot_usd['history'], height=150, color="#ffab00")
                
                st.markdown("#### 📅 Last 3 Weeks (Net Positioning)")
                # Create comparison table
                h_base = cot_base['raw_data'].head(3)[['Date', 'Long', 'Short', 'Net', 'OpenInt']].copy()
                h_base['Date'] = h_base['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(h_base, use_container_width=True, hide_index=True)

            else:
                st.info("COT Data not available yet for this asset.")

        with tab2:
            if seas:
                current_month = datetime.datetime.now().strftime('%B')
                st.markdown(f"#### 📅 Seasonal Pattern ({current_month})")
                
                c_seas1, c_seas2 = st.columns([1, 2])
                with c_seas1:
                    st.metric("Win Rate", f"{int(seas['win_rate'])}%")
                    st.caption(f"Historically, {sel_asset.split('/')[0]} has closed POSITIVE in {current_month} {int(seas['win_rate'])}% of the time over the last 10 years.")
                
                with c_seas2:
                    chart_df = pd.DataFrame({'Day': seas['chart'].index, 'Value': seas['chart'].values})
                    today_line = alt.Chart(pd.DataFrame({'x': [seas['day']]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
                    line = alt.Chart(chart_df).mark_line(color='#00e5ff').encode(
                        x=alt.X('Day', title='Day of Year'), 
                        y=alt.Y('Value', scale=alt.Scale(zero=False), title='Seasonality Index'),
                        tooltip=['Day', 'Value']
                    ).interactive()
                    st.altair_chart(line + today_line, use_container_width=True)
        
        with tab3:
            if sent["status"] == "OK":
                st.markdown("Retail Positioning")
                s_col, l_col = st.columns(2)
                s_col.metric("Short %", f"{sent['short']}%", delta_color="inverse")
                l_col.metric("Long %", f"{sent['long']}%", delta_color="normal")
                
                # Bar
                st.progress(sent['long']/100)
                if sent['long'] > 70: st.error("⚠️ Crowded LONG Trade (Contrarian Signal)")
                elif sent['short'] > 70: st.success("✅ Crowded SHORT Trade (Contrarian Signal)")
            else:
                st.warning("Sentiment data unavailable.")

    with col_right:
        st.markdown("#### 💪 Relative Strength")
        if pwr:
            st.write(f"**{sel_asset.split('/')[0]}** (Base)")
            st.progress(pwr['base'] / 10)
            st.caption(f"Score: {pwr['base']:.1f}/10 - {'Strong' if pwr['base']>6 else 'Weak'}")
            
            st.write(f"**USD** (Quote)")
            st.progress(pwr['usd'] / 10)
            st.caption(f"Score: {pwr['usd']:.1f}/10 - {'Strong' if pwr['usd']>6 else 'Weak'}")
        
        st.markdown("---")
        st.markdown("#### 🤖 AI Insight")
        if "gemini_response" in st.session_state:
            st.info(st.session_state["gemini_response"])
        else:
            st.caption("Go to AI Analyst tab to generate insights.")

elif selected == "AI Analyst":
    render_header("AI Analyst", "Powered by Gemini 1.5")
    
    # API KEY CHECK
    gemini_key = None
    try:
        gemini_key = st.secrets["GEMINI_KEY"]
    except:
        st.warning("⚠️ GEMINI_KEY missing in secrets.toml")
        gemini_key = st.text_input("Enter Gemini API Key manually", type="password")

    if gemini_key:
        mods = get_filtered_models(gemini_key)
        model_name = st.selectbox("Select Model", mods) if mods else "gemini-1.5-flash"
        
        user_q = st.chat_input("Ask about market conditions...")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            
            # Simple Context Preparation (Improve in future)
            context = f"Current Year: {datetime.datetime.now().year}. User asks: {user_q}"
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(context)
                        st.markdown(response.text)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        st.session_state["gemini_response"] = response.text[:200] + "..."
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Please configure API Key in settings or secrets.")

elif selected == "Settings":
    render_header("Settings")
    st.write("Configuration options here.")
