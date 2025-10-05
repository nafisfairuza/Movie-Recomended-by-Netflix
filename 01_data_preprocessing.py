import pandas as pd
import os
import numpy as np
from datetime import datetime

# --- 1. Konfigurasi ---
DATA_DIR = 'data'
RATING_COL = 'rating_imdb'
OUTPUT_FILE = 'merged_data.csv'

# Buat direktori data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- 2. Muat dan Bersihkan Data ---

# Definisikan mapping file ke tipe data
FILE_CONFIG = [
    # Data 1: IMDb movies.csv (Menggunakan 'avg_vote' sebagai kolom rating)
    {'filepath': 'IMDb movies.csv', 'type': 'movie', 'rating_col_name': 'avg_vote', 'title_col': 'title', 'year_col': 'year'}, # Asumsi: tambahkan 'year_col'
    # Data 2: Netflix Dataset.csv (Menggunakan 'IMDB Score' sebagai kolom rating)
    {'filepath': 'Netflix Dataset.csv', 'type': 'tv_show', 'rating_col_name': 'IMDB Score', 'title_col': 'Title', 'year_col': 'Release Date'}, # Asumsi: tambahkan 'year_col'
]

def load_and_clean_data(config):
    """Memuat, membersihkan, dan menstandarisasi kolom data."""
    filepath_full = os.path.join(DATA_DIR, config['filepath'])
    
    try:
        # Menggunakan encoding 'latin-1'
        df = pd.read_csv(filepath_full, encoding='latin-1', low_memory=False) 
    except FileNotFoundError:
        print(f"File {config['filepath']} tidak ditemukan. Lewati.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error saat memuat {config['filepath']}: {e}")
        return pd.DataFrame()

    print(f"Memproses {config['filepath']} ({len(df)} baris awal)...")
    
    # 0. Check for critical columns
    required_source_cols = [config['title_col'], config['rating_col_name']]
    if not all(col in df.columns for col in required_source_cols):
        missing = [col for col in required_source_cols if col not in df.columns]
        print(f"Peringatan: Kolom wajib ({', '.join(missing)}) hilang di {config['filepath']}. Lewati file ini.")
        return pd.DataFrame()

    df['type'] = config['type']
    
    # PERBAIKAN: Standarisasi nama kolom Title dan Rating
    df = df.rename(columns={
        config['title_col']: 'title', 
        config['rating_col_name']: RATING_COL
    })
    
    # 1. Kolom wajib
    df_filtered = df.copy()

    # 2. Standarisasi Kolom Opsional (Genre, Description, Director)
    
    # 2a. Genre
    genre_col = next((c for c in ['Genre', 'genre', 'listed_in'] if c in df.columns), None)
    if genre_col:
        df_filtered['genre'] = df[genre_col].astype(str).fillna('N/A')
    else:
        df_filtered['genre'] = 'N/A'
        
    # 2b. Description
    desc_col = next((c for c in ['description', 'Description', 'Plot', 'plot_summary'] if c in df.columns), None)
    if desc_col:
        df_filtered['description'] = df[desc_col].astype(str).fillna('No description provided')
    else:
        df_filtered['description'] = 'No description provided'

    # 2c. Director
    director_col = next((c for c in ['Director', 'director'] if c in df.columns), None)
    if director_col:
        df_filtered['director'] = df[director_col].astype(str).fillna('N/A')
    else:
        df_filtered['director'] = 'N/A'
    
    
    # --- KOREKSI STABILITAS DEPLOYMENT (PENTING) ---
    # Tambahkan Kolom Tahun dan pastikan itu string untuk RAG
    year_col = next((c for c in ['year', 'release_year', 'Release Date', 'date_published'] if c in df.columns), None)
    if year_col:
        # Jika kolom ada, ambil tahunnya (pastikan itu string)
        df_filtered['year'] = df[year_col].astype(str).apply(lambda x: str(x).split('-')[0] if '-' in str(x) else str(x)).replace('nan', 'N/A')
    else:
        df_filtered['year'] = 'N/A'
    # --- AKHIR KOREKSI STABILITAS ---
    
    
    # Kolom final (ditambahkan 'year' agar bisa masuk ke content_rag)
    final_cols = ['title', 'type', 'genre', 'director', RATING_COL, 'description', 'year']
    # Filter kolom yang benar-benar ada di DataFrame
    df_filtered = df_filtered[df_filtered.columns.intersection(final_cols)] 
    
    # Cleaning nilai NaN pada kolom kunci
    df_filtered = df_filtered.dropna(subset=['title', RATING_COL])
    
    # Konversi rating ke float
    df_filtered[RATING_COL] = pd.to_numeric(df_filtered[RATING_COL], errors='coerce')
    df_filtered = df_filtered.dropna(subset=[RATING_COL]) 

    return df_filtered


# Cleaning semua file yang dikonfigurasi
list_of_dfs = [load_and_clean_data(config) for config in FILE_CONFIG]

# Gabungkan data
final_df = pd.concat([df for df in list_of_dfs if not df.empty], ignore_index=True)
print(f"Total data gabungan setelah pembersihan: {len(final_df)}")

# --- 3. Feature Engineering untuk RAG ---

# Buat kolom content_rag (ditambahkan tahun)
final_df['content_rag'] = final_df.apply(
    lambda row: f"JUDUL: {row['title']} ({row.get('year', 'N/A')}) | TIPE: {row['type']} | GENRE: {row['genre']} | SUTRADARA: {row['director']} | RATING IMDB: {row[RATING_COL]} | PLOT: {row['description']}",
    axis=1
)

# --- 4. Batasi Data & Save ---

# BATASI DATA UNTUK MENGHEMAT KUOTA UNTUK DEMONSTRASI CEPAT
final_df = final_df.head(1000) # Batas dinaikkan ke 1000 untuk data RAG yang lebih baik.
print(f"Final Dataset Dibatasi untuk Demo: {len(final_df)} baris.")

output_path = os.path.join(DATA_DIR, OUTPUT_FILE)

final_df[['title', 'type', 'genre', 'director', RATING_COL, 'content_rag']].to_csv(output_path, index=False)

print(f"Data gabungan berhasil disimpan di: {output_path}")
