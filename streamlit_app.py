import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Fungsi ini akan di-cache oleh Streamlit, sehingga tidak perlu membuat ulang agent setiap kali ada interaksi
@st.cache_resource
def load_agent():
    load_dotenv()
    API = os.getenv('GOOGLE_API_KEY')
    if not API:
        st.error("API Key Google tidak ditemukan. Mohon atur di file .env Anda.")
        st.stop()

    # --- Inisialisasi semua komponen agent (sama seperti kode Anda sebelumnya) ---
    loader = PyPDFLoader("laporan.pdf")
    docs = loader.load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API, temperature=0)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    search_tool = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Pencari Dokumen Internal",
            func=rag_chain.run,
            description="Gunakan ini untuk menjawab pertanyaan spesifik tentang isi laporan dari PDF."
        ),
        Tool(
            name="Pencari Internet",
            func=search_tool.run,
            description="Gunakan ini untuk mencari informasi terkini, berita, atau topik umum di internet."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

# --- Tampilan Aplikasi Web ---
st.title("🤖 Agent Cerdas Honhon dan Beybey")
st.info("Tanya apa saja, baik tentang isi laporan PDF maupun informasi dari internet.")

# Muat agent yang sudah di-cache
agent = load_agent()

# Buat input teks untuk pengguna
user_question = st.text_input("Masukkan pertanyaan Anda:")

if user_question:
    with st.spinner("Agent sedang berpikir..."):
        # Jalankan agent dengan pertanyaan dari input web
        response = agent.invoke({"input": user_question})
        st.write("### Jawaban Agent:")
        st.success(response['output'])