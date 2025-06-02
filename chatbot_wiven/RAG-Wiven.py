import os
import json
import gradio as gr

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 🚀 nouveau package
from langchain_ollama import ChatOllama                  # 🚀 nouveau package
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────

JSON_DIR = "docs"  # 📁 Répertoire contenant vos fichiers .json

# Liste des fichiers disponibles
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
if not json_files:
    raise FileNotFoundError(f"Aucun fichier .json trouvé dans {JSON_DIR}.")

display_names = [f.removesuffix(".json") for f in json_files]
name2file = dict(zip(display_names, json_files))


# ────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ────────────────────────────────────────────────────────────────

def build_chain(json_file: str) -> RetrievalQA:
    """Construit (ou reconstruit) la chaîne QA pour le fichier demandé."""

    with open(os.path.join(JSON_DIR, json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mise en forme d’une vue ERP en texte libre
    def format_vue(d: dict) -> str:
        lines: list[str] = [f"Titre : {d.get('titre', '')}"]
        lines += ["\nApplications/Programmes :", *d.get("applications", [])]
        lines += ["\nFonctionnalités :", *d.get("fonctionnalites", [])]
        lines += ["\nBoutons :", *d.get("champs_et_boutons", {}).get("boutons", [])]
        lines += ["Cases à cocher :", *d.get("champs_et_boutons", {}).get("cases_a_cocher", [])]
        lines += ["\nDescription :", d.get("description", "")]
        return "\n".join(lines)

    # 1️⃣ Découpe le document en chunks
    doc = Document(page_content=format_vue(data))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])

    # 2️⃣ Vectorise les chunks avec un embedding HuggingFace en CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 3️⃣ LLM (= Mistral) via Ollama + Retriever
    llm = ChatOllama(model="mistral")
    retriever = vectorstore.as_retriever()

    # 4️⃣ Chaîne QA prête
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# ────────────────────────────────────────────────────────────────
# CALLBACKS GRADIO
# ────────────────────────────────────────────────────────────────

def init_chain(selected_name: str):
    """Callback → reconstruit la chaîne quand la vue change."""
    qa = build_chain(name2file[selected_name])
    return qa, gr.update(value="", interactive=True)


def ask_question(question: str, chat_history: list, qa_chain: RetrievalQA):
    """Envoie la question au modèle et renvoie la réponse."""
    if not question.strip():
        return gr.update(), chat_history

    answer = qa_chain.invoke(question)  # ← .invoke au lieu de .run
    chat_history.append((question, answer))
    return "", chat_history


# ────────────────────────────────────────────────────────────────
# INTERFACE GRADIO (widget intégrable)
# ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Chatbot ERP - Widget", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Chatbot ERP\nPosez vos questions sur la vue ERP sélectionnée.")

    vue_select = gr.Dropdown(
        choices=display_names,
        value=display_names[0],
        label="📂 Vue ERP",
    )

    # ⚠️ lambda → évite que Gradio appelle à tort RetrievalQA.__call__
    qa_chain_state = gr.State(lambda: build_chain(name2file[display_names[0]]))

    chatbot = gr.Chatbot(label="Historique")
    question_box = gr.Textbox(
        placeholder="Ex : Que fait le bouton Modifier ?",
        label="Votre question",
    )
    submit_btn = gr.Button("Envoyer", variant="primary")

    # 🖇️ Événements
    vue_select.change(init_chain, inputs=vue_select, outputs=[qa_chain_state, question_box])
    submit_btn.click(
        ask_question,
        inputs=[question_box, chatbot, qa_chain_state],
        outputs=[question_box, chatbot],
    )
    question_box.submit(
        ask_question,
        inputs=[question_box, chatbot, qa_chain_state],
        outputs=[question_box, chatbot],
    )


# ────────────────────────────────────────────────────────────────
# LANCEMENT
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
