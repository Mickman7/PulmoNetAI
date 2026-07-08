from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI

from ..database.db import init_db
from .routes import patients, predict, agent
import os


app = FastAPI(title="Pulmo API")


@app.on_event("startup")
def on_startup():
    init_db()  # creates tables if they don't exist yet


app.include_router(patients.router)
app.include_router(predict.router)
app.include_router(agent.router)


@app.get("/")
def root():
    return {"status": "Pulmo API running"}
