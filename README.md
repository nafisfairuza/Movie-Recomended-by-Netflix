# Movie and TV Show Recommender RAG Chatbot
This is a Retrieval-Augmented Generation (RAG) chatbot built using Streamlit and LangChain, designed to provide movie and TV show recommendations based on a custom dataset derived from Kaggle sources (merged_data.csv).
The RAG architecture ensures that recommendations are factual and grounded in the provided data, enhanced by the reasoning capabilities of an OpenAI Large Language Model (LLM).

1. Dataset Source The dataset used in this project comes from Kaggle. You can access it here: 🔗 https://www.kaggle.com/datasets/mirajshah07/netflix-dataset
  
2. Project Architecture & Components
Framework: Streamlit for the interactive web application interface.
Data Source: merged_data.csv (Contains information about movies/TV shows).
LLM (Online Deployment): OpenAI's GPT-3.5 Turbo (Requires an API Key).
Vector Store: ChromaDB (chroma_db/ folder) for storing vectorized movie data.
Embeddings: HuggingFace's all-MiniLM-L6-v2 (local embeddings used for indexing).

3. How to Run Locally
- Prerequisites
Python 3.8+ installed.
OpenAI API Key (for the online version).
Git installed (optional, for cloning).

-Setup
 * Clone the Repository:
git clone [https://github.com/nafisfairuza/Movie-Recommender-by-Netflix.git](https://github.com/nafisfairuza/Movie-Recommender-by-Netflix.git)
cd Movie-Recommender-by-Netflix
* Create and Activate Virtual Environment:
python -m venv venv
.\venv\Scripts\activate
* Install Dependencies:
pip install -r requirements.txt
* Configure API Key:
Create a file named .env in the root directory and add your key:
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
Note: The .env file is excluded from GitHub via the .gitignore.
* Build the Vector Store (Initial Setup)
This step processes merged_data.csv and creates the chroma_db/ folder.
python create_vectorstore.py
* Run the Streamlit App:
streamlit run app.py
The application will open in your web browser.

4. Deployment on Streamlit Cloud
This application is ready to be deployed on Streamlit Cloud, provided that the chroma_db/ folder is committed to this repository.
Configuration Steps:
a. Go to Streamlit Cloud and select this repository.
b. In the deployment settings, you must provide your OpenAI API Key as a secret.
c. Add the key/value pair in the Streamlit Cloud secrets configuration:
Key: OPENAI_API_KEY
Value: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (Your actual API Key)

The app will then load the committed chroma_db folder, install the dependencies, and run using the online OpenAI model.
