import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from github_loader import fetch_repo_files
from rag_engine import build_vectorstore, get_qa_chain

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# In-memory store per process (fine for a demo/free-tier single dyno)
VECTORSTORES = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")

@app.route("/connect", methods=["POST"])
def connect():
    repo_url = request.json.get("repo_url")
    files = fetch_repo_files(repo_url, GITHUB_TOKEN, max_files=40)
    if not files:
        return jsonify({"error": "No readable files found."}), 400
    try: 
        vs = build_vectorstore(files, repo_url) 
    except Exception as e: 
        return jsonify({"error": f"Failed to index repo: {str(e)}"}), 500 
    VECTORSTORES[repo_url] = vs
    session["repo"] = repo_url
    return jsonify({"message": f"Indexed {len(files)} files from {repo_url}"})



@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    repo_url = session.get("repo")
    vs = VECTORSTORES.get(repo_url)
    if not vs:
        return jsonify({"error": "Connect a repo first."}), 400
    qa = get_qa_chain(vs, GROQ_API_KEY)
    result = qa(question)
    sources = list({d.metadata["source"] for d in result["source_documents"]})
    return jsonify({"answer": result["result"], "sources": sources})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)