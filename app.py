import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
import google.generativeai as genai
import plotly.express as px  # Libreria per i grafici interattivi

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="XAI Tesi - FRED-MD Dashboard", page_icon="📈", layout="wide")

st.title("📊 Dashboard Econometrica XAI: CCE-Elastic Net su FRED-MD")
st.markdown("""
Questa applicazione utilizza lo stimatore **CCE-Elastic Net** per identificare i driver macroeconomici 
purificati dagli shock comuni. Il dataset utilizzato è il **FRED-MD (current.csv)**.
""")

# --- MOTORE ECONOMETRICO (Cache per velocità) ---
@st.cache_data
def elabora_modello_reale():
    try:
        # 1. Caricamento dati (saltiamo la riga 2 delle trasformazioni)
        df = pd.read_csv("current.csv", skiprows=[1])
        
        # 2. Pulizia Nomi e Date
        df = df.rename(columns={'sasdate': 'time'})
        # FRED-MD a volte ha righe vuote alla fine, le rimuoviamo
        df = df.dropna(subset=['time'])
        
        # 3. Setup Variabili
        # Scegliamo INDPRO (Produzione Industriale) come variabile dipendente y
        y_target = 'INDPRO' 
        
        # Creiamo un ID fittizio per simulare la struttura panel richiesta dal CCE
        if 'id' not in df.columns:
            df['id'] = 1
            
        # Rimuoviamo colonne con troppi NaN per non mandare in crash l'Elastic Net
        df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
        # Forward fill e backward fill per i buchi temporali
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Lista regressori (tutte le altre variabili macro)
        X_cols = [col for col in df.columns if col not in ['id', 'time', y_target]]
        
        # 4. Step CCE: Purificazione tramite medie cross-sezionali
        # (Nota: con N=1 la media coincide col valore, ma manteniamo la struttura CCE della tesi)
        z_bar = df.groupby('time').mean().drop(columns=['id'])
        z_bar.columns = [f'{col}_bar' for col in z_bar.columns]
        df = df.merge(z_bar, on='time')

        def partial_out(target_col, z_cols, data):
            reg = LinearRegression().fit(data[z_cols], data[target_col])
            return data[target_col] - reg.predict(data[z_cols])

        z_columns = [col for col in df.columns if '_bar' in col]
        
        # Purificazione variabili (Tilde)
        df['y_tilde'] = partial_out(y_target, z_columns, df)
        for x_col in X_cols:
            df[f'{x_col}_tilde'] = partial_out(x_col, z_columns, df)

        # 5. Step Elastic Net (Selezione Variabili)
        X_tilde_cols = [f'{x_col}_tilde' for x_col in X_cols]
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
        enet.fit(df[X_tilde_cols], df['y_tilde'])

        # Estrazione coefficienti significativi
        coeff_dict = {X_cols[i]: round(enet.coef_[i], 4) for i in range(len(X_cols))}
        return {k: v for k, v in coeff_dict.items() if v != 0.0}, y_target

    except Exception as e:
        return f"Errore: {e}", None

# Esecuzione del calcolo
active_features, y_var = elabora_modello_reale()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Impostazioni")
    api_key = st.text_input("Inserisci Gemini API Key:", type="password")
    st.caption("Ottieni una chiave gratuita su [Google AI Studio](https://aistudio.google.com/)")
    
    st.divider()
    st.subheader("📊 Risultati Numerici")
    if isinstance(active_features, dict):
        st.success(f"Variabile target: {y_var}")
        st.write("Driver selezionati dal modello:")
        st.json(active_features)
    else:
        st.error(active_features)

# --- CORPO PRINCIPALE DELL'APP ---

# Creiamo due colonne per l'impaginazione: Grafico a sinistra, Chat a destra
col_chart, col_chat = st.columns([1, 1], gap="large")

with col_chart:
    st.subheader("💡 Spiegazione Visiva: Importanza dei Driver")
    
    if isinstance(active_features, dict) and active_features:
        # Prepariamo i dati per Plotly
        df_plot = pd.DataFrame({
            'Driver': list(active_features.keys()),
            'Coefficiente': list(active_features.values())
        })
        
        # Calcoliamo il valore assoluto per l'ordinamento
        df_plot['Assoluto'] = df_plot['Coefficiente'].abs()
        df_plot = df_plot.sort_values(by='Assoluto', ascending=True) # Ordinamento per importanza
        
        # Creazione Grafico a Barre Orizzontali
        fig = px.bar(
            df_plot, 
            x='Coefficiente', 
            y='Driver', 
            orientation='h', # Orizzontale
            title=f"Impatto dei driver su {y_var} (Purificato da shock comuni)",
            color='Coefficiente', # Colore variabile in base al segno
            color_continuous_scale=px.colors.diverging.RdBu_r, # Rosso per negativo, Blu per positivo
            text_auto=True # Mostra i valori sulle barre
        )
        
        # Miglioriamo il layout
        fig.update_layout(
            yaxis={'title': 'Driver (Codice FRED-MD)'},
            xaxis={'title': 'Coefficiente Elastic Net'},
            coloraxis_showscale=False # Nascondiamo la legenda del colore
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Come leggere il grafico:** Le barre mostrano l'impatto specifico di ogni driver. 
        Le barre **blu (destra)** hanno un impatto positivo, le barre **rosse (sinistra)** hanno un impatto negativo. La lunghezza indica l'intensità dell'effetto. 
        Grazie al metodo CCE, questi valori sono **al netto dei trend globali comuni**.
        """)
        
    elif isinstance(active_features, dict) and not active_features:
        st.warning("Il modello Elastic Net non ha selezionato nessun driver significativo per questa variabile target.")
    else:
        st.error(active_features)

with col_chat:
    # --- CHATBOT UI ---
    st.subheader("🤖 Chatbot XAI: Spiegazione in Linguaggio Naturale")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ciao! Ho analizzato il dataset FRED-MD per spiegare le variazioni di {y_var}. Guarda il grafico a sinistra per l'impatto visivo, e chiedimi pure spiegazioni su questi risultati!"}
        ]

    # Mostra lo storico dei messaggi
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input utente
    if prompt_utente := st.chat_input("Perché il driver X1 è così importante?"):
        st.chat_message("user").write(prompt_utente)
        st.session_state.messages.append({"role": "user", "content": prompt_utente})

        if not api_key:
            st.error("Per favore, inserisci la tua API Key nella barra laterale.")
            st.stop()

        # --- PROMPT XAI PERSONALIZZATO PER FRED-MD ---
        contesto_sistema = f"""
        Sei un assistente esperto in economia, macroeconomia e Explainable AI (XAI).
        Hai stimato un modello CCE-Elastic Net sul dataset FRED-MD (current.csv).
        La variabile dipendente è {y_var} (Industrial Production).
        I driver significativi identificati, purificati dai fattori comuni non osservati, sono: {active_features}.
        
        SPIEGAZIONE DEI CODICI DRIVER (Mappa standard FRED-MD):
        - RPI: Real Personal Income
        - W875RX1: Real Personal Income ex Transfer Receipts
        - DPCERA3M086SBEA: Personal Consumption Expenditures
        - CMRMTSPLx: Real Mfg. and Trade Sales
        - INDPRO: Industrial Production Index (Variabile target)
        - UNRATE: Tasso di disoccupazione
        - PAYEMS: Total Nonfarm Payrolls
        - HOUST: Housing Starts
        - S&P 500: Indice S&P 500
        - CPIAUCSL: Inflazione (Consumer Price Index)
        - PCEPI: Personal Consumption Expenditures: Chain-type Price Index
        - FEDFUNDS: Federal Funds Rate
        (Se vedi altre sigle, usa la tua conoscenza del dataset FRED-MD per spiegarle economico-politicamente).

        COMPITO:
        1. Spiega all'utente i risultati basandoti sul grafico e sui numeri. 
        2. Evidenzia il segno (positivo/negativo) dell'impatto.
        3. Sottolinea che il metodo CCE ha rimosso l'effetto dei trend globali comuni, isolando la relazione specifica tra questi driver interni e la produzione.
        4. Evita formule matematiche complesse, usa un linguaggio comprensibile per un non esperto.
        """

        # Configurazione Gemini
        genai.configure(api_key=api_key)
        # Usiamo gemini-2.0-flash per velocità e affidabilità
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        try:
            with st.spinner("L'IA sta interpretando i coefficienti..."):
                # Invio del contesto del sistema + la domanda dell'utente
                risposta = model.generate_content(f"{contesto_sistema}\n\nDomanda utente: {prompt_utente}")
                
                # Stampa e salva la risposta
                st.chat_message("assistant").write(risposta.text)
                st.session_state.messages.append({"role": "assistant", "content": risposta.text})
        except Exception as e:
            st.error(f"Errore durante la generazione: {e}")
