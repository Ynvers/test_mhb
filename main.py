# Import libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
from io import BytesIO
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY')  # Assure-toi que ta clé API est correctement définie
print(f"API_KEY: {API_KEY}")
genai.configure(api_key=API_KEY)

# Initialiser le modèle
model = genai.GenerativeModel(model_name='gemini-1.5-pro')

# Définir l'application FastAPI
app = FastAPI()

# Définir le prompt que l'on envoie au modèle
prompt = """
Please analyze the central object in the provided image and determine whether it is a recyclable waste item.

Return your answer strictly in the following JSON format:
{
  "recyclable": true | false,
  "type": "string describing the category of waste (e.g., plastic, paper, metal, glass, organic, etc.) or null if not recyclable",
  "explanation": "a short and clear explanation for why the object is or is not recyclable, and how you identified the type"
}

Only consider the main object in the center of the image. Do not list multiple objects. Do not provide bounding boxes or any visual metadata.
"""

# Endpoint pour analyser l'image
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Charger l'image depuis le fichier
        img = Image.open(BytesIO(await file.read()))
        
        # Générer la réponse du modèle
        response = model.generate_content([
            img,
            (
                prompt
            ),
        ])
        
        # Récupérer le texte de la réponse du modèle
        result = response.text

        # Retourner la réponse sous forme de JSON
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
