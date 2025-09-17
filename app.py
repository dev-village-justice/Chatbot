from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin depuis votre site SPIP

# Charger la clé OpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Vérifier que la clé API est présente
if not api_key:
    logger.error("OPENAI_API_KEY non trouvée dans les variables d'environnement")
    raise ValueError("OPENAI_API_KEY manquante")

# Configuration globale du chain (initialisé une seule fois au démarrage)
qa_chain = None

def initialize_chain():
    """Initialise la chaîne de conversation une seule fois au démarrage"""
    global qa_chain
    
    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Tu es un assistant qui répond sur un programme d'événements juridiques.\n"
                "Analyse attentivement le CONTEXTE ci-dessous et réponds uniquement en fonction de ces données.\n\n"
                "Règles importantes :\n"
                "- Donne toujours l'heure (au format HH:MM) et le nom de la salle/emplacement dans ta réponse.\n"
                "- Si la question demande « premier », « dernier », « après-midi », ou un filtre similaire,\n"
                "  utilise les dates et heures pour trier ou sélectionner les bons événements.\n"
                "- Si aucun événement ne correspond, dis-le clairement, sans inventer.\n\n"
                "CONTEXTE :\n{context}\n\n"
                "QUESTION : {question}\n\n"
                "Réponse (avec heure et salle) :"
            ),
        )

        # Chemin vers le JSON
        base_dir = os.path.dirname(__file__)
        json_path = os.path.join(base_dir, "data", "sources.json")
        
        # Charger le fichier JSON
        with open(json_path, "r", encoding="utf-8") as f:
            event = json.load(f)

        # Préparer les documents
        docs = []
        
        # Chunk pour l'événement global
        global_chunk = f"""Nom de l'événement: {event['titre']}
Thème: {event['theme']}
Objectif: {event['objectif']}"""
        docs.append(Document(page_content=global_chunk))

        # Chunk pour chaque jour
        for jour_key in ["jour1", "jour2"]:
            jour_events = event["programme"].get(jour_key, [])
            if jour_events:
                content = f"{jour_key.upper()} : "
                for e in jour_events:
                    content += (
                        f"{e['date']} {e['heures']} - {e['emplacement']} - {e['sujet']}. "
                    )
                docs.append(Document(page_content=content))

        # Découpage en chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        # Embeddings et vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(split_docs, embeddings)

        # Chaîne de conversation
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=vectorstore.as_retriever(),
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        logger.info("Chaîne de conversation initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la chaîne: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Point de santé pour vérifier que le service fonctionne"""
    return jsonify({"status": "healthy", "message": "Service LLM opérationnel"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal pour les questions au chatbot"""
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Question manquante", 
                "message": "Veuillez fournir une question"
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                "error": "Question vide", 
                "message": "La question ne peut pas être vide"
            }), 400
        
        # Historique de conversation (optionnel)
        chat_history = data.get('chat_history', [])
        
        # Appeler le modèle
        result = qa_chain({"question": question, "chat_history": chat_history})
        answer = result['answer']
        
        logger.info(f"Question: {question}")
        logger.info(f"Réponse: {answer}")
        
        return jsonify({
            "answer": answer,
            "question": question,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {str(e)}")
        return jsonify({
            "error": "Erreur interne", 
            "message": "Une erreur s'est produite lors du traitement de votre question"
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec documentation simple"""
    return jsonify({
        "service": "LLM Chatbot API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Vérification de santé du service",
            "/chat": "POST - Poser une question au chatbot"
        },
        "usage": {
            "chat_endpoint": {
                "method": "POST",
                "body": {
                    "question": "Votre question ici",
                    "chat_history": "Historique optionnel"
                }
            }
        }
    })

if __name__ == '__main__':
    # Initialiser la chaîne au démarrage
    initialize_chain()
    
    # Démarrer le serveur Flask
    # Pour la production, utilisez un serveur WSGI comme Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)