@echo off
echo.
echo ----------------------------------------
echo  INSTALLATION DES DEPENDANCES ERP CHATBOT
echo ----------------------------------------
echo.

REM Vérifier que pip est installé
where pip >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERREUR : pip n'est pas installé ou pas dans le PATH.
    pause
    exit /b 1
)

REM Mise à jour de pip (optionnel)
echo 📦 Mise à jour de pip...
pip install --upgrade pip

REM Installation des bibliothèques principales
echo 📦 Installation de langchain, streamlit, faiss, sentence-transformers...
pip install -U langchain langchain-community langchain-openai streamlit faiss-cpu sentence-transformers python-dotenv

echo.
echo ✅ INSTALLATION TERMINEE
pause
