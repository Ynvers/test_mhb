# Import libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
from io import BytesIO
import os
import json


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

Renvoyez votre réponse au format JSON suivant :
{
"recyclable": true | false,
"type": "chaîne décrivant la catégorie de déchet (par exemple, plastique, papier, métal, verre, matière organique, etc.) ou null si non recyclable",
"explanation": "une explication courte et claire expliquant pourquoi l'objet est recyclable ou non, et comment vous avez identifié le type",
"quantity": "Une chaine représentant le poids de l'objet (par exemple petit, moyen, grand) ou null si non recyclable",
"kwetche": "une nombre représentant la quantitité de points de recyclage de l'objet (par exemple, 0-100) basé sur son poids. Par exemple un objet de 1kg rapporte 100 points de recyclage, un objet de 0.5kg rapporte 50 points de recyclage, etc.",
}

Considérez uniquement l'objet principal au centre de l'image. N'en listez pas plusieurs. Ne fournissez pas de cadres de délimitation ni de métadonnées visuelles.
"""

# Endpoint pour analyser l'image
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Charger l'image depuis le fichier
        img = Image.open(BytesIO(await file.read()))
        
        # Générer la réponse du modèle
        response = model.generate_content([img, prompt])
        
        # Récupérer le texte de la réponse du modèle
        result = response.text

        # Parse la réponse texte en JSON Python
        result_json = json.loads(response.text)
        
        # Retourner la réponse sous forme de JSON
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
