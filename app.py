import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
import google.generativeai as genai

# --- 1. CONFIGURAZIONE PAGINA E UI ---
st.set_page_config(page_title="XAI CCE-ElasticNet", page_icon="🤖")
st.title("📊 Chatbot XAI: CCE-Elastic Net")
st.markdown("Interroga i risultati del modello econometrico in linguaggio naturale.")

# --- 2. MOTORE ECONOMETRICO (Eseguito una sola volta grazie alla cache) ---
@st.cache_data
def esegui_stima_cce_en():
    np.random.seed(42)
    N, T, K = 50, 30, 20
    f_t = np.random.normal(0, 1, T)
    data = []
    for i in range(N):
        gamma_i = np.random.uniform(0.5, 1.5)
        lambda_i = np.random.uniform(0.2, 0.8, K)
        for t in range(T):
            X_it = lambda_i * f_t[t] + np.random.normal(0, 1, K)
            # Veri driver: X1, X2, X3
            y_it = 2.0*X_it[0] - 1.5*X_it[1] + 1.0*X_it[2] + gamma_i*f_t[t] + np.random.normal(0, 0.5)
            data.append([i, t, y_it] + list(X_it))
            
    cols = ['id', 'time', 'y'] + [f'X_{k+1}' for k in range(K)]
    df = pd.DataFrame(data, columns=cols)

    z_bar = df.groupby('time').mean().drop(columns=['id'])
    z_bar.columns = [f'{col}_bar' for col in z_bar.columns]
    df = df.merge(z_bar, on='time')

    def partial_out(target_col, z_cols, data):
        reg = LinearRegression().fit(data[z_cols], data[target_col])
        return data[target_col] - reg.predict(data[z_cols])

    z_columns = [col for col in df.columns if '_bar' in col]
    df['y_tilde'] = partial_out('y', z_columns, df)
    X_cols = [f'X_{k+1}' for k in range(K)]
    for x_col in X_cols:
        df[f'{x_col}_tilde'] = partial_out(x_col, z_columns, df)

    X_tilde_cols = [f'{x_col}_tilde' for x_col in X_cols]
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
    enet.fit(df[X_tilde_cols], df['y_tilde'])

    coeff_dict = {X_cols[i]: round(enet.coef_[i], 3) for i in range(K)}
    return {k: v for k, v in coeff_dict.items() if v != 0.0}

active_features = esegui_stima_cce_en()

# --- 3. SIDEBAR (Per la sicurezza e i dettagli tecnici) ---
with st.sidebar:
    st.header("⚙️ Impostazioni")
    # Inserimento chiave API sicuro (non rimane salvata nel codice!)
    api_key = st.text_input("Inserisci la tua API Key di Gemini:", type="password")
    
    st.divider()
    st.subheader("📈 Risultati CCE-Elastic Net")
    st.caption("Variabili purificate dai fattori comuni:")
    st.json(active_features)

# --- 4. LOGICA CHATBOT LLM ---
if "messages" not in st.session_state:
    # Messaggio di benvenuto iniziale
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Ho analizzato i dati con il modello CCE-Elastic Net purificandoli dai fattori comuni. Chiedimi pure cosa vuoi sapere sulla performance aziendale!"}]

# Mostra lo storico della chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 5. INTERAZIONE UTENTE ---
if prompt_utente := st.chat_input("Chiedi qualcosa (es. 'Qual è la variabile più importante?')"):
    
    # 1. Stampa il messaggio dell'utente a schermo
    st.chat_message("user").write(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})

    # 2. Controllo API Key
    if not api_key:
        st.error("AIzaSyCx7mzeTZQJb1Kvd8GAwcaDetsimLTvQSo")
        st.stop()

    # 3. Costruzione del Contesto "Nascosto" per l'IA
    contesto_sistema = f"""
    Sei un assistente XAI (Explainable AI). Hai appena stimato un modello CCE-Elastic Net 
    su dati panel. I risultati (al netto degli shock globali non osservati) indicano che 
    solo queste variabili hanno un impatto: {active_features}.
    Tutte le altre 17 variabili sono irrilevanti (coefficiente 0).
    Rispondi in modo professionale ma comprensibile, basandoti SOLO su questi numeri. 
    L'utente chiede: {prompt_utente}
    """

    # 4. Chiamata a Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    with st.spinner("Sto analizzando i dati econometrici..."):
        try:
            risposta = model.generate_content(contesto_sistema)
            testo_risposta = risposta.text
            
            # Stampa la risposta del bot e salvala nello storico
            st.chat_message("assistant").write(testo_risposta)
            st.session_state.messages.append({"role": "assistant", "content": testo_risposta})
        except Exception as e:
            st.error(f"Errore API: {e}")
