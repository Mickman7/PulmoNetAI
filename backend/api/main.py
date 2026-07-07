import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Load variables from .env file before initializing routes or models
load_dotenv()

app = FastAPI(title="PulmoNetAI API")

# Include your routes below...