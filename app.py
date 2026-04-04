import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
import google.generativeai as genai
import plotly.express as px

# --- SETUP PAGINA ---
st.set_page_config(page_title="Tesi Saveria Falvo - Dashboard XAI", layout="wide")

# --- BARRA LATERALE (Spostata in alto per essere letta subito) ---
with st.sidebar:
    st.header("⚙️ Impostazioni")
    # Usiamo una chiave univoca 'chiave_gemini' per evitare che si cancelli
    api_key = st.text_input("Inserisci Gemini API Key:", type="password", key="chiave_gemini")
    st.caption("Ottieni una chiave gratuita su Google AI Studio")
    st.divider()

st.title("📊 Framework Econometrico: CCE-Elastic Net")
st.caption("Analisi empirica su dataset FRED-MD - Allineato allo sviluppo in R")

# --- MOTORE ECONOMETRICO (Cache) ---
@st.cache_data
def esegui_stima_empirica():
    try:
        df = pd.read_csv("current.csv")
        df = df.drop(0).reset_index(drop=True)
        df = df.rename(columns={df.columns[0]: 'time'}).dropna(subset=['time'])
        y_target = 'INDPRO'
        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, thresh=int(0.8 * len(df)))
        df_numeric = df_numeric.ffill().bfill()
        
        Y = df_numeric[[y_target]].values
        X = df_numeric.drop(columns=[y_target]).values
        feature_names = df_numeric.drop(columns=[y_target]).columns.tolist()
        T_obs, K_vars = X.shape

        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        Sigma_T = (1/T_obs) * (X_std @ X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_T)
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
        
        m_hat = int(np.sum(eigenvalues >= (0.05 * eigenvalues[0])))
        if m_hat == 0: m_hat = 1

        F_hat = eigenvectors[:, :m_hat]
        M_hat = np.eye(T_obs) - F_hat @ np.linalg.inv(F_hat.T @ F_hat) @ F_hat.T
        
        Y_proj, X_proj = M_hat @ Y, M_hat @ X
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=2000)
        model.fit(X_proj, Y_proj.ravel())

        coeffs = {feature_names[i]: round(model.coef_[i], 5) for i in range(K_vars) if abs(model.coef_[i]) > 1e-4}
        return coeffs, y_target, m_hat, model.l1_ratio_, model.alpha_
    except Exception as e:
        return f"Errore tecnico: {str(e)}", None, None, None, None

active_features, target, factors, opt_alpha, opt_lambda = esegui_stima_empirica()

# --- LAYOUT DASHBOARD ---
col_stats, col_chat = st.columns([1, 1], gap="large")

with col_stats:
    st.subheader("📈 Risultati della Stima")
    if isinstance(active_features, dict):
        st.write(f"**Variabile Target:** {target}")
        st.metric("Fattori comuni (m_hat)", factors)
        if active_features:
            df_plot = pd.DataFrame({'Variabile': list(active_features.keys()), 'Coefficiente': list(active_features.values())}).sort_values(by='Coefficiente')
            fig = px.bar(df_plot, x='Coefficiente', y='Variabile', orientation='h', color='Coefficiente', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(active_features)

with col_chat:
    st.subheader("🤖 Chatbot XAI")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Analisi completata. Inserisci l'API Key a sinistra e chiedimi pure spiegazioni!"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt_utente := st.chat_input("Fai una domanda sui driver..."):
        st.chat_message("user").write(prompt_utente)
        st.session_state.messages.append({"role": "user", "content": prompt_utente})
        
        # Recuperiamo la chiave dallo stato o dall'input
        chiave_finale = st.session_state.get('chiave_gemini', api_key)

        if not chiave_finale:
            st.error("⚠️ Chiave API non rilevata. Inseriscila nella barra laterale e premi INVIO prima di scrivere in chat.")
            st.stop()
        
        try:
            genai.configure(api_key=chiave_finale)
            model_llm = genai.GenerativeModel('gemini-2.5-flash')
            prompt_xai = f"Sei un esperto di econometria. Risultati: Driver={active_features}, Target={target}, m_hat={factors}. Spiega all'utente: {prompt_utente}"
            
            with st.spinner("L'IA sta elaborando i dati..."):
                res = model_llm.generate_content(prompt_xai)
                st.chat_message("assistant").write(res.text)
                st.session_state.messages.append({"role": "assistant", "content": res.text})
        except Exception as e:
            st.error(f"Errore API: {e}")
