import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # <<< EMBEDDING LOKAL (GRATIS)
from langchain_openai import ChatOpenAI # <<< CHAT MODEL API KEY
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import time 

# --- 1. Konfigurasi Awal ---
load_dotenv()

api_key_value = os.getenv("OPENAI_API_KEY")

if not api_key_value:
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
def get_rag_chain(_vectorstore):
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
    
    # Inisialisasi Chat Model OpenAI
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
    st.set_page_config(page_title="Rekomendasi Film by Netflix", layout="centered")
    
    st.title("🎬 Chatbot Rekomendasi Film & TV")
    st.markdown("_Tanyakan pada saya tentang rekomendasi film terpopuler saat ini._")
    
    vectorstore = load_vector_store()
    if vectorstore is None:
        return 

    # Inisialisasi RAG Chain
    qa_chain = get_rag_chain(vectorstore) 
    
    # Inisialisasi session state untuk menyimpan riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya Roger, asisten rekomendasi film terbaik. Apa yang ingin kamu cari?"}]
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
            with st.spinner("Wait a second, jangan pergi dulu..."):
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
