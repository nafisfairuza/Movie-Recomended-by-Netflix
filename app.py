'''
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma # Sudah diperbaiki
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.dataframe import DataFrameLoader
import time # Tambahkan ini untuk simulasi loading

# --- 1. Konfigurasi Awal ---
load_dotenv()

# Ambil nilai kunci API dari file .env
api_key_value = os.getenv("GEMINI_API_KEY")

if api_key_value:
    os.environ["GOOGLE_API_KEY"] = api_key_value
    os.environ["GEMINI_API_KEY"] = api_key_value
else:
    st.error("GEMINI_API_KEY tidak ditemukan. Pastikan file .env sudah diisi dan berada di folder yang sama.")
    st.stop()

CHROMA_PATH = "./chroma_db"

# --- 2. Inisialisasi Vector Store dan Model ---

@st.cache_resource
def load_and_process_data():
    """Memuat Vector Store yang sudah ada. Jika belum ada, coba buat baru."""
    
    key = os.getenv("GOOGLE_API_KEY") 
    
    # 2a. Inisialisasi Embeddings (diperlukan untuk memuat atau membuat)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=key 
    )

    # 2b. Cek apakah Vector Store sudah ada
    # Chroma menyimpan file internalnya; jika direktori ada dan memiliki isi, berarti sudah tersimpan.
    is_persisted = os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 1
    
    if is_persisted:
        st.success("Vector Store berhasil dimuat dari data tersimpan.")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            return vectorstore
        except Exception as e:
            st.error(f"Gagal memuat Vector Store tersimpan: {e}. Coba hapus folder 'chroma_db' dan jalankan 'create_vectorstore.py' lagi.")
            return None
    
    # 2c. Jika belum ada (Saat pertama kali jalan / setelah dihapus)
    st.warning("Vector Store belum tersedia. Membuat Vector Store baru dari data mentah.")
    
    # Memuat Data (Hanya jika perlu membuat baru)
    data_path = 'data/merged_data.csv'
    try:
        df = pd.read_csv(data_path)
        loader = DataFrameLoader(df, page_content_column="content_rag")
        documents = loader.load()
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan: {data_path}. Pastikan '01_data_preprocessing.py' sudah dijalankan.")
        return None
    
    # Membuat Vector Store
    try:
        with st.spinner("Membuat embeddings (Membutuhkan Kuota API, ini hanya perlu sekali)..."):
            # Simulasi waktu embedding untuk UX
            time.sleep(2) 
            vectorstore = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=CHROMA_PATH
            )
        st.success("Vector Store berhasil dibuat dan disimpan.")
        return vectorstore
    except Exception as e:
        # Jika GAGAL (karena Quota 429), tampilkan pesan
        st.error(f"Gagal membuat Vector Store (Periksa Kuota API Anda): {e}. Jalankan 'create_vectorstore.py' secara terpisah!")
        return None


# --- 3. Inisialisasi RAG Chain ---

@st.cache_resource
def get_rag_chain(vectorstore):
    """Menginisialisasi Model dan RAG Chain."""
    
    template = """
    Anda adalah asisten rekomendasi film yang ramah dan berpengetahuan.
    Gunakan potongan konteks yang diambil berikut ini untuk menjawab pertanyaan pengguna.
    Jika Anda tidak dapat menemukan jawaban dari konteks, katakan bahwa Anda tidak memiliki informasi tersebut dan berikan saran umum untuk menonton film.
    
    Konteks: {context}
    Pertanyaan: {question}
    Jawaban:
    """
    
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=key 
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), 
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False 
    )
    
    return qa_chain

# --- 4. Fungsi Utama Chatbot ---

def main():
    st.set_page_config(page_title="Rekomendasi Film RAG Gemini", layout="centered")
    
    st.title("🎬 Chatbot Rekomendasi Film & TV RAG dengan Gemini")
    st.markdown("_Tanyakan pada saya tentang rekomendasi film, rating IMDB, atau informasi plot._")
    
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY tidak ditemukan. Pastikan file .env sudah diisi.")
        return

    # Muat data dan vector store
    vectorstore = load_and_process_data()
    if vectorstore is None:
        return 

    # Inisialisasi RAG Chain
    qa_chain = get_rag_chain(vectorstore)
    
    # Inisialisasi session state untuk menyimpan riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya asisten rekomendasi film Anda. Tanyakan tentang film berdasarkan genre, rating, atau aktor!"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Tampilkan riwayat chat sebelumnya
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dari pengguna
    if prompt := st.chat_input("Tanyakan rekomendasi film atau informasi IMDB..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses permintaan dan dapatkan respons
        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari rekomendasi..."):
                try:
                    result = qa_chain.invoke(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    response = result["answer"]
                    
                    st.session_state.chat_history.append((prompt, response))
                    
                except Exception as e:
                    response = f"Terjadi kesalahan saat memproses permintaan: {e}. Mohon coba lagi."
                    print(f"RAG Chain Error: {e}") 

            st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
'''

'''
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # <<< EMBEDDING LOKAL (GRATIS)
from langchain_openai import ChatOpenAI # <<< CHAT MODEL API (ONLINE, BUTUH KUOTA BARU)
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import time 

# --- 1. Konfigurasi Awal ---
load_dotenv()

# Ambil nilai kunci API OpenAI untuk LLM Chat (Membaca API Key baru dari .env)
api_key_value = os.getenv("OPENAI_API_KEY")

if not api_key_value:
    # Cek API Key
    st.error("OPENAI_API_KEY tidak ditemukan di file .env. Pastikan Anda sudah memasukkan kunci API baru.")
    st.stop()

CHROMA_PATH = "./chroma_db"

# --- 2. Inisialisasi Vector Store dan Model ---

@st.cache_resource
def load_vector_store():
    """Memuat Vector Store yang sudah ada dengan fungsi Embedding yang tepat (Lokal)."""
    
    # 2a. Inisialisasi Embeddings LOKAL (HuggingFace)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Gagal memuat model Embedding lokal. Error: {e}")
        return None

    # 2b. Cek apakah Vector Store sudah ada
    is_persisted = os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 1
    
    if is_persisted:
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            return vectorstore
        except Exception as e:
            st.error(f"Gagal memuat Vector Store: {e}.")
            return None
    
    st.error("Vector Store belum tersedia. Silakan jalankan 'python create_vectorstore.py' di terminal terlebih dahulu.")
    return None

# --- 3. Inisialisasi RAG Chain ---

@st.cache_resource
def get_rag_chain(_vectorstore): # Menggunakan _vectorstore untuk menghindari error caching Streamlit
    """Menginisialisasi Model dan RAG Chain."""
    
    template = """
    Anda adalah asisten rekomendasi film yang ramah dan berpengetahuan.
    Gunakan potongan konteks yang diambil berikut ini untuk menjawab pertanyaan pengguna.
    Jika Anda tidak dapat menemukan jawaban dari konteks, katakan bahwa Anda tidak memiliki informasi tersebut dan berikan saran umum untuk menonton film.
    
    Konteks: {context}
    Pertanyaan: {question}
    Jawaban:
    """
    
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    # Inisialisasi Chat Model OpenAI (Menggunakan API Key baru Anda)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.3,
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}), 
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False 
    )
    
    return qa_chain

# --- 4. Fungsi Utama Chatbot ---

def main():
    st.set_page_config(page_title="Rekomendasi Film RAG Lokal", layout="centered")
    
    st.title("🎬 Chatbot Rekomendasi Film & TV RAG (Lokal & OpenAI)")
    st.markdown("_Tanyakan pada saya tentang rekomendasi film, rating IMDB, atau informasi plot._")
    
    # Muat data dan vector store
    vectorstore = load_vector_store()
    if vectorstore is None:
        return 

    # Inisialisasi RAG Chain
    qa_chain = get_rag_chain(vectorstore) 
    
    # Inisialisasi session state untuk menyimpan riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya asisten rekomendasi film Anda. Tanyakan tentang film berdasarkan genre, rating, atau aktor!"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Tampilkan riwayat chat sebelumnya
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dari pengguna
    if prompt := st.chat_input("Tanyakan rekomendasi film atau informasi IMDB..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses permintaan dan dapatkan respons
        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari rekomendasi..."):
                try:
                    result = qa_chain.invoke(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    response = result["answer"]
                    
                    st.session_state.chat_history.append((prompt, response))
                    
                except Exception as e:
                    # Error akan muncul di sini jika API Key yang baru masih belum berkuota
                    response = f"Terjadi kesalahan saat memproses permintaan LLM. (Cek Kuota API OpenAI Anda). Error: {e}"
                    print(f"RAG Chain Error: {e}") 

            st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
'''


import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # EMBEDDING LOKAL (GRATIS)
from langchain_community.chat_models import ChatOllama # <<< CHAT MODEL LOKAL (GEMMA 2B)
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import time 

# --- 1. Konfigurasi Awal ---
load_dotenv()
CHROMA_PATH = "./chroma_db"

# --- 2. Inisialisasi Vector Store dan Model ---

@st.cache_resource
def load_vector_store():
    """Memuat Vector Store yang sudah ada dengan fungsi Embedding yang tepat (Lokal)."""
    
    # 2a. Inisialisasi Embeddings LOKAL (HuggingFace)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Gagal memuat model Embedding lokal. Error: {e}")
        return None

    # 2b. Cek apakah Vector Store sudah ada
    is_persisted = os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 1
    
    if is_persisted:
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            return vectorstore
        except Exception as e:
            st.error(f"Gagal memuat Vector Store: {e}.")
            return None
    
    st.error("Vector Store belum tersedia. Silakan jalankan 'python create_vectorstore.py' di terminal terlebih dahulu.")
    return None

# --- 3. Inisialisasi RAG Chain ---

@st.cache_resource
def get_rag_chain(_vectorstore): 
    """Menginisialisasi Model dan RAG Chain."""
    
    template = """
    Anda adalah asisten rekomendasi film yang ramah dan berpengetahuan.
    Gunakan potongan konteks yang diambil berikut ini (data film/TV) untuk menjawab pertanyaan pengguna.
    Jika Anda tidak dapat menemukan jawaban dari konteks (data film), katakan bahwa Anda tidak memiliki informasi tersebut dan berikan saran umum untuk menonton film.
    
    Konteks: {context}
    Pertanyaan: {question}
    Jawaban:
    """
    
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    # Inisialisasi Chat Model LOKAL (OLLAMA) DENGAN GEMMA 2B
    llm = ChatOllama(
        model="gemma:2b", # GANTI DARI llama3 KE gemma:2b
        temperature=0.3,
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}), 
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False 
    )
    
    return qa_chain

# --- 4. Fungsi Utama Chatbot ---

def main():
    st.set_page_config(page_title="Rekomendasi Film RAG LOKAL", layout="centered")
    
    st.title("🎬 Chatbot Rekomendasi Film & TV RAG (Ollama LOKAL)")
    st.markdown("_Tanyakan pada saya tentang rekomendasi film, rating IMDB, atau informasi plot._")
    
    # Muat data dan vector store
    vectorstore = load_vector_store()
    if vectorstore is None:
        return 

    # Inisialisasi RAG Chain
    qa_chain = get_rag_chain(vectorstore) 
    
    # Inisialisasi session state untuk menyimpan riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya asisten rekomendasi film Anda. Tanyakan tentang film berdasarkan genre, rating, atau aktor!"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Tampilkan riwayat chat sebelumnya
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dari pengguna
    if prompt := st.chat_input("Tanyakan rekomendasi film atau informasi IMDB..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses permintaan dan dapatkan respons
        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari rekomendasi dengan LLM LOKAL..."):
                try:
                    result = qa_chain.invoke(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    response = result["answer"]
                    
                    st.session_state.chat_history.append((prompt, response))
                    
                except Exception as e:
                    # Error akan muncul di sini jika Ollama tidak berjalan
                    response = f"Terjadi kesalahan saat memproses permintaan LLM. Pastikan **Ollama sedang berjalan** di *background* dan model **gemma:2b** sudah terunduh. Error: {e}"
                    print(f"RAG Chain Error: {e}") 

            st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
