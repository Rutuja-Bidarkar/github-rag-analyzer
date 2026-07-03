import os
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_repo_cache_path(repo_url):
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()
    return f"cache/{repo_hash}"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(files, repo_url):
    cache_path = get_repo_cache_path(repo_url)
    embeddings = get_embeddings()

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