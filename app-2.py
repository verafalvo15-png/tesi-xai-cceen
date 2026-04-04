import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
import google.generativeai as genai
import plotly.express as px

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Tesi Saveria Falvo - XAI Dashboard", page_icon="🤖", layout="wide")
st.title("📊 Framework Econometrico: CCE-Elastic Net")
st.markdown("Analisi empirica basata sul dataset **FRED-MD** (current.csv) con purificazione CSD.")

# --- 2. MOTORE ECONOMETRICO (Allineato al tuo codice R) ---
@st.cache_data
def esegui_stima_empirica():
    try:
        # Caricamento dati saltando la riga delle trasformazioni
        df = pd.read_csv("current.csv", skiprows=[1])
        df = df.rename(columns={'sasdate': 'time'}).dropna(subset=['time'])
        
        # Target: INDPRO (Produzione Industriale)
        y_target = 'INDPRO'
        
        # Pulizia dati (na.omit)
        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, thresh=int(0.9 * len(df)))
        df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
        
        Y = df_numeric[[y_target]].values
        X = df_numeric.drop(columns=[y_target]).values
        feature_names = df_numeric.drop(columns=[y_target]).columns.tolist()
        
        T_obs, K_vars = X.shape

        # --- STEP PCA (Identificazione Fattori Comuni m_hat) ---
        # Sigma_T = (1/T) * X * X'
        Sigma_T = (1/T_obs) * (X @ X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_T)
        # Ordine decrescente
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Criterio 5% del primo autovalore (Tua logica R)
        m_hat = np.sum(eigenvalues >= (0.05 * eigenvalues[0]))
        
        # --- STEP PROIEZIONE ORTOGONALE CCE ---
        F_hat = eigenvectors[:, :m_hat]
        I_T = np.eye(T_obs)
        # Matrice di proiezione M = I - F(F'F)^-1 F'
        M_hat = I_T - F_hat @ np.linalg.inv(F_hat.T @ F_hat) @ F_hat.T
        
        # Purificazione (Tilde)
        Y_proj = M_hat @ Y
        X_proj = M_hat @ X

        # --- STEP ELASTIC NET CV (Tuning Alpha e Lambda) ---
        # L1_ratio tra 0.1 e 1 (Lasso) per gestire il Grouping Effect
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10, selection='random')
        model.fit(X_proj, Y_proj.ravel())

        # Estrazione coefficienti non nulli
        active_coeffs = {feature_names[i]: round(model.coef_[i], 5) for i in range(K_vars) if model.coef_[i] != 0}
        
        return active_coeffs, y_target, m_hat, model.l1_ratio_, model.alpha_

    except Exception as e:
        return str(e), None, None, None, None

# Esecuzione analisi
active_features, target, factors, opt_alpha, opt_lambda = esegui_stima_empirica()

# --- 3. LAYOUT DASHBOARD ---
col_stats, col_chat = st.columns([1, 1], gap="large")

with col_stats:
    st.subheader("📈 Risultati dell'Analisi")
    if isinstance(active_features, dict):
        # Metriche principali
        m1, m2, m3 = st.columns(3)
        m1.metric("Fattori rilevati (m)", factors)
        m2.metric("Alpha EN (Opt)", opt_alpha)
        m3.metric("Target", target)
        
        # Grafico a barre interattivo
        df_plot = pd.DataFrame({
            'Variabile': list(active_features.keys()),
            'Coefficiente': list(active_features.values())
        }).sort_values(by='Coefficiente', ascending=True)
        
        fig = px.bar(df_plot, x='Coefficiente', y='Variabile', orientation='h',
                     title="Importanza Driver (Dati Purificati CCE)",
                     color='Coefficiente', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Errore: {active_features}")

with col_chat:
    st.subheader("🤖 Chatbot XAI")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ciao Saveria! Ho analizzato il dataset FRED-MD applicando la proiezione ortogonale. Chiedimi pure come interpretare i driver selezionati!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if p_utente := st.chat_input("Spiegami perché questi driver sono significativi..."):
        st.chat_message("user").write(p_utente)
        st.session_state.messages.append({"role": "user", "content": p_utente})
        
        # API Key dalla Sidebar
        api_key_input = st.sidebar.text_input("Inserisci Gemini API Key:", type="password")
        
        if not api_key_input:
            st.warning("Inserisci la chiave API nella barra laterale a sinistra per attivare il chatbot.")
            st.stop()
            
        genai.configure(api_key=api_key_input)
        model_llm = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt istruito con la tua metodologia
        contesto_sistema = f"""
        Sei il chatbot della tesi di Saveria Falvo. 
        Hai analizzato il dataset FRED-MD usando lo stimatore CCE-Elastic Net.
        
        DETTAGLI METODOLOGICI:
        - Purificazione CSD tramite Proiezione Ortogonale (M_hat) basata su PCA.
        - Numero fattori comuni (m_hat): {factors}.
        - Alpha ottimale (Elastic Net): {opt_alpha}.
        - Variabili selezionate: {active_features}.
        
        RISPONDI ALL'UTENTE:
        Spiega l'impatto economico delle variabili selezionate. 
        Sottolinea che grazie al metodo CCE abbiamo rimosso i trend macroeconomici comuni (shock globali), 
        isolando la relazione specifica tra i driver e la produzione industriale.
        """
        
        with st.spinner("L'IA sta elaborando i risultati..."):
            try:
                response = model_llm.generate_content(f"{contesto_sistema}\n\nDomanda utente: {prompt_utente}")
                st.chat_message("assistant").write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Errore: {e}")
