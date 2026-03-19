import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if key is loaded
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OPENAI_API_KEY in a .env file")

