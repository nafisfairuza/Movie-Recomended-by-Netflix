'''
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.dataframe import DataFrameLoader

# --- Konfigurasi ---
load_dotenv()
CHROMA_PATH = "./chroma_db"
DATA_PATH = 'data/merged_data.csv'

# Ambil nilai kunci API dari file .env
api_key_value = os.getenv("GEMINI_API_KEY")

if api_key_value:
    os.environ["GOOGLE_API_KEY"] = api_key_value
else:
    print("ERROR: GEMINI_API_KEY tidak ditemukan. Pastikan file .env sudah diisi.")
    exit()

def create_vector_store():
    """Membuat Vector Store baru dan menyimpannya ke disk."""
    
    # 1. Muat Data
    try:
        df = pd.read_csv(DATA_PATH)
        loader = DataFrameLoader(df, page_content_column="content_rag")
        documents = loader.load()
        print(f"Memuat {len(documents)} dokumen dari {DATA_PATH}...")
    except FileNotFoundError:
        print(f"ERROR: File data tidak ditemukan: {DATA_PATH}. Jalankan '01_data_preprocessing.py' terlebih dahulu.")
        return
    
    # 2. Inisialisasi Embeddings
    key = os.getenv("GOOGLE_API_KEY") 
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=key 
    )
    
    # 3. Membuat Vector Store
    print(f"Membuat Vector Store dan menyimpan ke {CHROMA_PATH}. Ini mungkin memakan waktu...")
    try:
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        print("SUCCESS! Vector Store berhasil dibuat dan disimpan ke disk.")
    except Exception as e:
        print(f"FATAL ERROR: Gagal membuat Vector Store.")
        print(f"Penyebab: {e}")
        print("Pastikan GEMINI_API_KEY Anda aktif dan kuota *embedding* belum habis (Error 429).")


if __name__ == "__main__":
    # Bersihkan folder lama (opsional, tapi disarankan)
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Folder lama {CHROMA_PATH} berhasil dibersihkan.")
        
    create_vector_store()
    '''

'''
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings # MENGGUNAKAN OPENAI
from langchain_community.document_loaders.dataframe import DataFrameLoader
import shutil

# --- Konfigurasi ---
load_dotenv()
CHROMA_PATH = "./chroma_db"
DATA_PATH = 'data/merged_data.csv'

# Ambil nilai kunci API dari file .env
api_key_value = os.getenv("OPENAI_API_KEY")

if not api_key_value:
    print("ERROR: OPENAI_API_KEY tidak ditemukan. Pastikan file .env sudah diisi.")
    exit()

def create_vector_store():
    """Membuat Vector Store baru dan menyimpannya ke disk."""
    
    # 1. Muat Data
    try:
        df = pd.read_csv(DATA_PATH)
        loader = DataFrameLoader(df, page_content_column="content_rag")
        documents = loader.load()
        print(f"Memuat {len(documents)} dokumen dari {DATA_PATH}...")
    except FileNotFoundError:
        print(f"ERROR: File data tidak ditemukan: {DATA_PATH}. Jalankan '01_data_preprocessing.py' terlebih dahulu.")
        return
    
    # 2. Inisialisasi Embeddings OpenAI
    # Model default untuk embedding OpenAI adalah text-embedding-ada-002
    embeddings = OpenAIEmbeddings()
    
    # 3. Membuat Vector Store
    print(f"Membuat Vector Store dan menyimpan ke {CHROMA_PATH}. Ini mungkin memakan waktu...")
    try:
        # Hapus folder lama
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Folder lama {CHROMA_PATH} berhasil dibersihkan.")

        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        print("SUCCESS! Vector Store berhasil dibuat dan disimpan ke disk.")
    except Exception as e:
        print(f"FATAL ERROR: Gagal membuat Vector Store.")
        print(f"Penyebab: {e}")
        print("Pastikan OPENAI_API_KEY Anda valid dan memiliki kuota.")


if __name__ == "__main__":
    create_vector_store()
    '''


import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # <<< KITA PAKAI INI!
from langchain_community.document_loaders.dataframe import DataFrameLoader
import shutil

# --- Konfigurasi ---
load_dotenv()
CHROMA_PATH = "./chroma_db"
DATA_PATH = 'data/merged_data.csv'

# KAMI TIDAK MEMERLUKAN API KEY APAPUN UNTUK FILE INI LAGI.

def create_vector_store():
    """Membuat Vector Store baru dan menyimpannya ke disk secara lokal."""
    
    # 1. Muat Data
    try:
        df = pd.read_csv(DATA_PATH)
        loader = DataFrameLoader(df, page_content_column="content_rag")
        documents = loader.load()
        print(f"Memuat {len(documents)} dokumen dari {DATA_PATH}...")
    except FileNotFoundError:
        print(f"ERROR: File data tidak ditemukan: {DATA_PATH}. Jalankan '01_data_preprocessing.py' terlebih dahulu.")
        return
    
    # 2. Inisialisasi Embeddings LOKAL (HuggingFace)
    print("Menggunakan HuggingFace Embeddings (LOKAL) - TIDAK BUTUH KUOTA API.")
    # Model default yang cepat dan stabil: all-MiniLM-L6-v2
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        print("ERROR: Gagal menginisialisasi HuggingFace Embeddings.")
        print("Pastikan Anda sudah menjalankan: pip install sentence-transformers")
        return
    
    # 3. Membuat Vector Store
    print(f"Membuat Vector Store dan menyimpan ke {CHROMA_PATH}. Ini mungkin memakan waktu...")
    try:
        # Hapus folder lama
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Folder lama {CHROMA_PATH} berhasil dibersihkan.")

        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        print("SUCCESS! Vector Store berhasil dibuat dan disimpan ke disk.")
    except Exception as e:
        print(f"FATAL ERROR: Gagal membuat Vector Store.")
        print(f"Penyebab: {e}")
        print("Meskipun tidak butuh API Key, mungkin ada masalah izin folder atau memori.")


if __name__ == "__main__":
    create_vector_store()