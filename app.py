from datetime import datetime
from src.script.use_script import UseScript
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random, json, os

use = UseScript("./api/model/sentiment_model_tfidf.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


class InputData(BaseModel):
    username: str
    bio: str
    marketing_emails: bool

DATA_FILE = "data.json"

def generate_unique_id(existing_ids):
    """Genera un ID univoco a 6 cifre che non è già presente."""
    while True:
        new_id = f"{random.randint(100000, 999999)}"
        if new_id not in existing_ids:
            return new_id

@app.post("/save-data/")
async def save_data(data: InputData):
    # Legge il contenuto esistente del file JSON (se presente)
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Estrae gli ID esistenti per evitare duplicati
    existing_ids = {entry["id"] for entry in existing_data}

    # Genera un ID univoco
    unique_id = generate_unique_id(existing_ids)

    # Crea un dizionario per i dati ricevuti, includendo l'ID, data, numero_commenti e commenti
    data_with_id = {
        "id": unique_id,
        "username": data.username,
        "bio": data.bio,
        "marketing_emails": data.marketing_emails,
        "status": "Active",
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Data di salvataggio nel formato "YYYY-MM-DD HH:MM:SS"
        "numero_commenti": 0,  # Numero di commenti, inizialmente 0
        "commenti": {},  # Commenti vuoti inizialmente
    }

    # Aggiunge i nuovi dati a quelli esistenti
    existing_data.append(data_with_id)

    # Scrive tutti i dati nel file JSON
    with open(DATA_FILE, "w") as file:
        json.dump(existing_data, file, indent=4)

    return {"message": "Data saved successfully", "id": unique_id}



@app.get("/get-data/{item_id}")
async def get_data(item_id: int):
    # Controlla se il file JSON esiste
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Data file not found")

    # Legge i dati dal file JSON
    with open(DATA_FILE, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Data file is corrupted")

    # Cerca l'elemento con l'ID corrispondente
    for entry in data:
        if entry.get("id") == str(item_id):  # Confronta l'ID come stringa
            return {"message": "Data found", "data": entry}

    # Se l'ID non è trovato, solleva un errore
    raise HTTPException(status_code=404, detail="Item not found")




# Struttura del commento da ricevere
class CommentRequest(BaseModel):
    id: str
    comment: str

# Funzione per caricare i dati dal file JSON esistente (se presente)
def load_data():
    try:
        with open('data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []




# Endpoint per aggiungere un commento
@app.post("/add_comment/")
async def add_comment(request: CommentRequest):
    # Carica i dati esistenti dal file
    data = load_data()

    # Cerca l'utente nel file usando l'id
    for user in data:
        if user['id'] == request.id:

            c_id = str(len(user['commenti']) + 1)

            print(use.use_model(request.comment))

            new_comment = {
                "comento": request.comment,
                "sentimento": use.use_model(request.comment),
            }

            # Aggiungi il nuovo commento alla lista
            user['commenti'][c_id] = new_comment

            # Aggiorna il numero dei commenti
            user['numero_commenti'] = str(len(user['commenti']))
            break

    # Salva i dati aggiornati nel file
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)

    return {"message": "Commento aggiunto con successo!"}




@app.get("/get-data/")
async def get_data():
    # Controlla se il file JSON esiste
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Data file not found")

    # Legge i dati dal file JSON
    with open(DATA_FILE, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Data file is corrupted")

    # Restituisce tutti i dati
    return {"message": "Data retrieved successfully", "data": data}