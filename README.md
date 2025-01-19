# machine_learning_project

### Introduzione al progetto

TeachTuner è un’applicazione progettata per applicare tecniche di analisi del sentiment nel contesto educativo. Lo scopo principale è automatizzare la raccolta e l’elaborazione dei feedback degli studenti, classificandoli in sentiment positivi, negativi o neutri. Ciò consente agli educatori di ottenere informazioni utili per migliorare la qualità dell’insegnamento e ottimizzare le strategie didattiche.

**Caratteristiche principali**:

- Raccolta di commenti tramite form personalizzati.
- Preprocessing e normalizzazione dei dati testuali.
- Utilizzo di modelli di machine learning e deep learning per l’analisi del sentiment.
- Presentazione dei risultati tramite report visuali e metriche.

**Tecnologie Utilizzate**:

- Backend sviluppato in Python e supportato da FastAPI. 
- Frontend basato su Vue.js per una gestione interattiva. 
- Pipeline di machine learning conforme agli standard CRISP-DM.

### Istruzioni per l’esecuzione

Seguire i passaggi riportati per configurare e avviare il progetto:

Clona il repository:

    git clone git@github.com:Pr1vat30/machine_learning_project.git
    cd machine_learning_project

Configura ambiente python:

    pyenv local 3.12.7
    python3.12 -m venv .venv
    source .venv/bin/activate

Installa le dipendenze Backend:

    pip install -r ./src/utils/requirement.txt

Avvia il server FastAPI:

    python -m uvicorn app:app --reload --port 8080

Installa dipendeze Frontend:

    cd ui/my-vue-app
    npm install

Avvia il server di sviluppo di Vue.js:
    
    npm run dev
    go to http://localhost:5173/