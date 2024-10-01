import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("WEAVEL_TESTING") == "true":
    ENDPOINT_URL = "http://localhost:8000"
else:
    ENDPOINT_URL = "https://api.weavel.ai"
