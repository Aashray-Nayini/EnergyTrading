from .entsoe_data import EntsoeData
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env into environment variables
API_KEY_ENTSOE = os.getenv("API_KEY_ENTSOE")
