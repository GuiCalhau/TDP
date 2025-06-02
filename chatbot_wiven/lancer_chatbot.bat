@echo off
echo.
echo ----------------------------------------
echo  LANCEMENT DU CHATBOT ERP - WIVEN.AI
echo ----------------------------------------
echo.

REM Aller dans le dossier du projet
# cd /d P:\chatbot_erp_wiven

REM Lancer l’application Streamlit sur le réseau local
echo Demarrage de l'application Streamlit...
# streamlit run app.py --server.address 192.168.1.129 --server.port 8501
streamlit run app2.py

pause
