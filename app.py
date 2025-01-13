from datetime import datetime
from src.script.use_script import UseScript
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random, json, os

DATA_FILE = "data.json"

use = UseScript("./api/model/deployed_model.pkl")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    activity_name: str
    description: str
    allow_visibility: bool

class CommentRequest(BaseModel):
    id: str
    comment: str

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
        "activity_name": data.activity_name,
        "description": data.description,
        "visibility": data.allow_visibility,
        "status": "Active",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_comment": 0,
        "comments": {},
    }

    # Aggiunge i nuovi dati a quelli esistenti
    existing_data.append(data_with_id)

    # Scrive tutti i dati nel file JSON
    with open(DATA_FILE, "w") as file:
        json.dump(existing_data, file, indent=4)

    return {"message": "Data saved successfully", "id": unique_id}

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

@app.post("/add_comment/")
async def add_comment(request: CommentRequest):
    # Carica i dati esistenti dal file
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Cerca l'utente nel file usando l'id
    for entry in data:
        if entry['id'] == request.id:

            c_id = str(len(entry['comments']) + 1)

            print(use.use_model(request.comment))

            new_comment = {
                "comment": request.comment,
                "sentiment": use.use_model(request.comment),
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            # Aggiungi il nuovo commento alla lista
            entry['comments'][c_id] = new_comment

            # Aggiorna il numero dei commenti
            entry['n_comment'] = str(len(entry['comments']))
            break

    # Salva i dati aggiornati nel file
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)

    return {"message": "Commento aggiunto con successo!"}




