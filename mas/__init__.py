import os
# from dotenv import load_dotenv
# load_dotenv()
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

# Only set environment variables if values are present to avoid TypeError
if api_base is not None:
    os.environ["OPENAI_API_BASE"] = api_base
if api_key is not None:
    os.environ["OPENAI_API_KEY"] = api_key