import os
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_repo_cache_path(repo_url):
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()
    return f"cache/{repo_hash}"


def get_embeddings(cohere_api_key):
    return CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=cohere_api_key
    )


def build_vectorstore(files, repo_url, cohere_api_key):
    cache_path = get_repo_cache_path(repo_url)
    embeddings = get_embeddings(cohere_api_key)

    if os.path.exists(cache_path):
        print(f"Loading cached index for {repo_url}")
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)

    print(f"No cache found — indexing {repo_url} fresh")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    documents = []
    for f in files:
        chunks = splitter.split_text(f["content"])
        for c in chunks:
            documents.append(Document(page_content=c, metadata={"source": f["path"]}))

    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs("cache", exist_ok=True)
    vectorstore.save_local(cache_path)
    print(f"Indexed and cached {len(documents)} chunks for {repo_url}")
    return vectorstore


def get_qa_chain(vectorstore, groq_api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_qa(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {"result": answer, "source_documents": docs}

    return run_qa

def get_overview_cache_path(repo_url):
    normalized = repo_url.strip().rstrip("/").lower()
    repo_hash = hashlib.md5(normalized.encode()).hexdigest()
    return f"cache/{repo_hash}_overview.json"


def generate_repo_overview(files, repo_url, groq_api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.1
    )

    file_paths = [f["path"] for f in files if f["path"] != "__REPO_STRUCTURE__"]
    tree_text = "\n".join(sorted(file_paths))

    priority_names = {
        "readme.md", "package.json", "requirements.txt", "pyproject.toml",
        "app.py", "main.py", "server.js", "index.js", "index.ts",
        "manage.py", "settings.py", "tsconfig.json", "next.config.js",
        "vite.config.js", "dockerfile", "docker-compose.yml"
    }

    key_snippets = []
    for f in files:
        name_lower = f["path"].split("/")[-1].lower()
        if name_lower in priority_names:
            snippet = f["content"][:1500]
            key_snippets.append(f"--- {f['path']} ---\n{snippet}")

    key_content = "\n\n".join(key_snippets[:8])

    prompt = ChatPromptTemplate.from_template("""
You are analyzing a GitHub repository to generate a structured technical overview.

Repository file tree:
{tree}

Key file contents:
{key_content}

Based on the above, respond with ONLY a valid JSON object (no markdown, no code fences, no explanation) with exactly these keys:

{{
  "summary": "2-3 sentence description of what this project does",
  "key_features": ["3-5 short bullet points of standout features"],
  "tech_stack": ["list", "of", "detected", "technologies"],
  "architecture_style": "short description e.g. MVC, component-based, REST API, etc.",
  "entry_points": ["main files where execution starts"],
  "folder_overview": [
    {{"folder": "folder/path", "purpose": "short description"}}
  ],
  "main_files": [
    {{"file": "path/to/file", "why": "why it's important to look at first"}}
  ],
  "dependencies": ["key libraries/frameworks detected"],
  "data_flow": "short description of how a request/data flows through the app",
  "setup_instructions": ["inferred step-by-step commands to run this project locally"]
}}

Only output the JSON object, nothing else.
""")

    chain = prompt | llm
    response = chain.invoke({"tree": tree_text, "key_content": key_content})
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        overview = json.loads(raw)
    except json.JSONDecodeError:
        overview = {
            "summary": raw, "key_features": [], "tech_stack": [], "architecture_style": "",
            "entry_points": [], "folder_overview": [], "main_files": [],
            "dependencies": [], "data_flow": "", "setup_instructions": []
        }

    return overview


def get_or_generate_overview(files, repo_url, groq_api_key):
    cache_path = get_overview_cache_path(repo_url)

    if os.path.exists(cache_path):
        print(f"Loading cached overview for {repo_url}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Generating fresh overview for {repo_url}")
    overview = generate_repo_overview(files, repo_url, groq_api_key)

    os.makedirs("cache", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(overview, f)

    return overview