import streamlit as st
import altair as alt
import pandas as pd

def render_metric_card(col, label, value, delta=None, color="normal"):
    """
    Render a custom styled metric card.
    """
    col.metric(label, value, delta)

def render_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")

def render_seasonal_chart(seas_data):
    if not seas_data:
        st.warning("Dati stagionali non disponibili")
        return

    st.markdown("##### Ciclo Annuale (Median Path)")
    chart_df = pd.DataFrame({'Giorno': seas_data['chart'].index, 'Valore': seas_data['chart'].values})
    
    # Current Day Line
    today_line = alt.Chart(pd.DataFrame({'x': [seas_data['day']]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
    
    # Main Line
    line = alt.Chart(chart_df).mark_line(color='#00e5ff').encode(
        x=alt.X('Giorno', title="Giorno dell'anno"), 
        y=alt.Y('Valore', scale=alt.Scale(zero=False), title='Index (100)'),
        tooltip=['Giorno', 'Valore']
    ).properties(height=300)
    
    st.altair_chart(line + today_line, use_container_width=True)

def render_cot_chart(cot_data, color):
    if not cot_data:
        return
    st.area_chart(cot_data['history'], height=150, color=color)

def display_full_analysis(tech, cot_base, cot_usd, sent, seas, sel_asset):
    # This function organizes the display logic that was previously in the main script
    
    # Top Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    if tech:
        c1.metric("PREZZO", f"{tech['price']:.4f}", f"{tech['change']:.2f}%")
        c2.metric("RSI (14)", f"{tech['rsi']:.1f}")
        c3.metric("MACD Hist", f"{tech['macd_hist']:.4f}", delta="Positivo" if tech['macd_hist']>0 else "Negativo")
        c4.metric("TREND (MA200)", tech['trend_desc'])
        c5.metric("VOLATILITÀ", f"{tech['atr']:.4f}")
    
    st.markdown("---")
