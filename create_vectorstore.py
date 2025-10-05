import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # <<< Menggunakan Embeddings LOKAL (GRATIS)
from langchain_community.document_loaders.dataframe import DataFrameLoader
import shutil

# --- Konfigurasi ---
load_dotenv()
CHROMA_PATH = "./chroma_db"
DATA_PATH = 'data/merged_data.csv'

# Catatan: File ini TIDAK membutuhkan API Key karena menggunakan model lokal.

def create_vector_store():
    """Membuat Vector Store baru dan menyimpannya ke disk secara lokal menggunakan model gratis."""
    
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
    # Model: all-MiniLM-L6-v2 (Cepat, efisien, dan stabil untuk RAG)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        print("ERROR: Gagal menginisialisasi HuggingFace Embeddings.")
        print(f"Penyebab: {e}")
        print("Pastikan Anda sudah menginstal dependency: pip install sentence-transformers")
        return
    
    # 3. Membuat Vector Store
    print(f"Membuat Vector Store dan menyimpan ke {CHROMA_PATH}. Ini mungkin memakan waktu...")
    try:
        # Hapus folder lama untuk memastikan kebaruan data
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
