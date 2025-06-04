# Wiven.ai – Assistant ERP basé sur l’IA (Travail de diplôme)

**Auteur :** Guilherme De Oliveira Calhau  
**Entreprise :** Mediasoft & Cie S.A.  
**Formation :** Technicien ES – CPNV  
**Date :** Juin 2025

## 🎯 Objectif du projet

Ce projet vise à créer un prototype d’assistant intelligent capable de répondre à des questions liées à un ERP (Enterprise Resource Planning), notamment dans les domaines **comptable** et **fonctionnel**. Le prototype s’intègre dans le projet global **Wiven.ai** de l’entreprise Mediasoft & Cie S.A.

## 🛠️ Technologies utilisées

- **LangChain** – Orchestration du pipeline RAG
- **FAISS** – Indexation vectorielle locale
- **HuggingFace** – Embeddings (`all-MiniLM-L6-v2`) et intégration LLM
- **Ollama** – Exécution locale de Mistral 7B
- **Streamlit** – Interface utilisateur simple pour le chatbot
- **Python** – Backend général
- **Format JSON** – Structuration des vues ERP

## 🧠 Approche technique

### 1. 🔍 Analyse des LLM
Comparaison de plusieurs modèles (Mistral, DeepSeek, LLaMA3, etc.) sur des questions comptables, en français et anglais.

### 2. 🔬 Tentative d'ajustement (Fine-tuning via QLoRA)
- Résultats peu satisfaisants à cause des limites matérielles et de la taille réduite des données.
- Abandon du fine-tuning au profit d’une approche plus efficace et locale.

### 3. 🧪 Test avec IBM Watsonx.ai
Évaluation rapide d’une alternative cloud SaaS. Solution non retenue en raison d’un manque de contrôle local.

### 4. 🧩 Réorientation vers RAG
Mise en place d’un système **Retrieval-Augmented Generation** avec accès direct à une base documentaire ERP.

#### Fonctionnement :
- Chargement d’un fichier JSON décrivant une vue ERP.
- Vectorisation avec HuggingFace.
- Recherche dans un index FAISS.
- Génération de réponse par Mistral.
- Interface utilisateur via Streamlit.



## 🗂️ Structure du dépôt


```
📁 projet/
├── 📁 chatbot_wiven/
│   ├── 📁 vues_json/
│   │   ├── ba126.json    #Page ba126 en json pour le RAG
│   │   ├── ba127.json    #Page ba127 en json pour le RAG
│   │   └── rm100.json    #Page rm100 en json pour le RAG
│   ├── .env
│   ├── CFM token.txt
│   ├── Guide_utilisateur_chatbot_WivenAI.docx
│   ├── Guide_utilisateur_chatbot_WivenAI.pdf
│   ├── install_requirements.bat     # installe la packege nécessaire 
│   ├── lancer_chatbot.bat     # Lance le script RAG-Wiven.py dans un fichier
│   ├── RAG-Wiven.py    #script pour le RAG
│   ├── requirements.txt
│   ├── wiven_01.jpg
│   └── wiven_02.jpg 
├── 📁 QLoRA/
│   ├── augmented_plan_comptable.jsonl    # Le plan comptable en JSONL pour ajuster LLM
│   ├── Données de base.docx
│   ├── ExemplePlanComptable.xlsx    # Le plan comptable de base 
│   ├── Salaire.docx      
│   └── train_qlora.py    # Script pour entrainer LLM
├── plan-initial.png    # Planning initial du projet
└── plan-final.png     # Planning final du projet
```
