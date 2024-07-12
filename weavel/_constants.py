import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("WEAVEL_TESTING") == "true":
    BACKEND_SERVER_URL = "http://localhost:8000"
else:
    BACKEND_SERVER_URL = "https://unravel.up.railway.app"