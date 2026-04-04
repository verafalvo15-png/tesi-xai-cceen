import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
import google.generativeai as genai
import plotly.express as px

# --- SETUP PAGINA ---
st.set_page_config(page_title="Tesi Saveria Falvo - Dashboard XAI", layout="wide")
st.title("📊 Framework Econometrico: CCE-Elastic Net")
st.caption("Analisi empirica su dataset FRED-MD - Allineato allo sviluppo in R")

# --- MOTORE ECONOMETRICO (Versione Robusta) ---
@st.cache_data
def esegui_stima_empirica():
    try:
        # 1. Caricamento Dati: saltiamo la riga 2 (trasformazioni)
        df = pd.read_csv("current.csv")
        # Rimuoviamo la riga delle trasformazioni (la riga 0 dopo il caricamento)
        df = df.drop(0).reset_index(drop=True)
        
        # Pulizia colonna temporale
        df = df.rename(columns={df.columns[0]: 'time'})
        df = df.dropna(subset=['time'])
        
        # Selezione Target (INDPRO)
        y_target = 'INDPRO'
        
        # Teniamo solo colonne numeriche e rimuoviamo quelle con troppi NaN
        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, thresh=int(0.8 * len(df)))
        # Riempimento buchi (Imputation)
        df_numeric = df_numeric.ffill().bfill()
        
        if y_target not in df_numeric.columns:
            return f"Errore: Variabile {y_target} non trovata nel dataset.", None, None, None, None

        Y = df_numeric[[y_target]].values
        X = df_numeric.drop(columns=[y_target]).values
        feature_names = df_numeric.drop(columns=[y_target]).columns.tolist()
        
        T_obs, K_vars = X.shape

        # 2. PCA PER FATTORI COMUNI (Metodo R)
        # Standardizzazione (Fondamentale per PCA)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        Sigma_T = (1/T_obs) * (X_std @ X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_T)
        
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Soglia 5% autovalore principale
        m_hat = int(np.sum(eigenvalues >= (0.05 * eigenvalues[0])))
        if m_hat == 0: m_hat = 1

        # 3. PROIEZIONE ORTOGONALE M_hat
        F_hat = eigenvectors[:, :m_hat]
        I_T = np.eye(T_obs)
        M_hat = I_T - F_hat @ np.linalg.inv(F_hat.T @ F_hat) @ F_hat.T
        
        # Purificazione
        Y_proj = M_hat @ Y
        X_proj = M_hat @ X

        # 4. ELASTIC NET CV
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=2000)
        model.fit(X_proj, Y_proj.ravel())

        coeffs = {feature_names[i]: round(model.coef_[i], 5) for i in range(K_vars) if abs(model.coef_[i]) > 1e-4}
        
        return coeffs, y_target, m_hat, model.l1_ratio_, model.alpha_

    except Exception as e:
        return f"Errore tecnico: {str(e)}", None, None, None, None

# Esecuzione
active_features, target, factors, opt_alpha, opt_lambda = esegui_stima_empirica()

# --- INTERFACCIA ---
col_stats, col_chat = st.columns([1, 1], gap="large")

with col_stats:
    st.subheader("📈 Risultati della Stima")
    if isinstance(active_features, dict):
        st.write(f"**Variabile Target:** {target}")
        st.metric("Fattori comuni (m_hat)", factors)
        
        if active_features:
            df_plot = pd.DataFrame({
                'Variabile': list(active_features.keys()),
                'Coefficiente': list(active_features.values())
            }).sort_values(by='Coefficiente')
            
            fig = px.bar(df_plot, x='Coefficiente', y='Variabile', orientation='h',
                         color='Coefficiente', color_continuous_scale='RdBu_r',
                         title="Impatto dei Driver (Purificati CCE)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nessuna variabile selezionata dall'Elastic Net con i parametri attuali.")
    else:
        st.error(active_features)

with col_chat:
    st.subheader("🤖 Chatbot XAI")
    # (Codice Chatbot identico al precedente...)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Analisi completata. Chiedimi pure spiegazioni sui driver macroeconomici!"}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if p_utente := st.chat_input("Spiegami i risultati..."):
        st.chat_message("user").write(p_utente)
        st.session_state.messages.append({"role": "user", "content": p_utente})
        api_key = st.sidebar.text_input("Gemini API Key", type="password")
        if not api_key:
            st.warning("Inserisci l'API Key nella barra laterale!")
            st.stop()
        genai.configure(api_key=api_key)
        model_llm = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Sei un esperto XAI. Target: {target}, Fattori: {factors}, Driver: {active_features}. Spiega all'utente: {p_utente}"
        with st.spinner("Pensando..."):
            res = model_llm.generate_content(prompt)
            st.chat_message("assistant").write(res.text)
            st.session_state.messages.append({"role": "assistant", "content": res.text})
