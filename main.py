# Import libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
from io import BytesIO
import os
import json
import re


# Configuration
API_KEY = os.environ.get('GEMINI_API_KEY')  # Assure-toi que ta clé API est correctement définie
print(f"API_KEY: {API_KEY}")
genai.configure(api_key=API_KEY)

# Initialiser le modèle
model = genai.GenerativeModel(model_name='gemini-1.5-pro')

# Définir l'application FastAPI
app = FastAPI()

# Définir le prompt que l'on envoie au modèle
prompt = """
Veuillez analyser l’objet principal au centre de l’image fournie et déterminer s’il s’agit d’un déchet recyclable.
Si c’est le cas, classez-le selon l’une des catégories suivantes :
    - Plastique : bouteilles, flacons, pots de yaourt, barquettes, sacs, films plastiques…
    - Papier/Carton : journaux, feuilles, cahiers, boîtes d’emballage, cartons…
    - Métal : canettes, boîtes de conserve, barquettes en aluminium, capsules, aérosols vides…
    - Verre : bouteilles, pots, bocaux, flacons…
   
Si l’objet n’est pas dans la liste, indiquez "recyclable": false

TU DOIS RÉPONDRE UNIQUEMENT AU FORMAT JSON VALIDE. 
Analyse l'objet principal dans l'image et renvoie CE SCHÉMA EXACT :

{
  "recyclable": true|false,
  "type": "plastique|papier|métal|verre|null",
  "explanation": "2-3 phrases max",
  "quantity": "petit|moyen|grand|null",
  "kwetche": "nombre entre 0 et 100"
}

Exemple de réponse valide :
```json
{
  "recyclable": true,
  "type": "plastique",
  "explanation": "Bouteille en PET recyclable",
  "quantity": "moyen",
  "kwetche": 50
}
Considérez uniquement l'objet principal au centre de l'image. N'en listez pas plusieurs. Ne fournissez pas de cadres de délimitation ni de métadonnées visuelles.
"""

def clean_gemini_response(text: str) -> dict:
    """Nettoie la réponse de Gemini pour extraire le JSON valide."""
    try:
        # Supprime les marqueurs ```json et autres artefacts
        cleaned = re.sub(r'```json|```', '', text).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Tentative de récupération du JSON même si mal formaté
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        raise

# Endpoint pour analyser l'image
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Seules les images sont acceptées"}
            )

        img = Image.open(BytesIO(await file.read()))
        response = model.generate_content([img, prompt])
        
        if not response.text:
            return JSONResponse(
                status_code=500,
                content={"error": "Réponse vide de l'API Gemini"}
            )

        # Nettoyage et validation de la réponse
        result = clean_gemini_response(response.text)
        
        # Retourne directement l'objet JSON parsé
        return result

    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Réponse JSON invalide de Gemini",
                "details": str(e),
                "raw_response": response.text if 'response' in locals() else None
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur interne: {str(e)}"}
        )