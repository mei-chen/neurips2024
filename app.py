import streamlit as st
import json
from typing import List, Dict
import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from datetime import datetime
import time
import pickle
import urllib.parse

class EmbeddingProcessor:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize the embedding processor with OpenAI credentials."""
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using OpenAI's API with retry logic."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def search_similar(self, query: str, embedding_data: Dict, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        query_embedding = self.get_embedding(query)
        
        similarities = np.dot(embedding_data['embeddings'], query_embedding) / (
            np.linalg.norm(embedding_data['embeddings'], axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = embedding_data['documents'][idx]
            result = {
                'poster_number': doc['poster_number'],
                'title': doc['title'],
                'abstract': doc['abstract'],
                'authors': doc['authors'],
                'similarity': float(similarities[idx]),
                'session_info': {
                    'session_name': doc['session_info']['session_name'],
                    'location': doc['session_info']['location'],
                    'time': doc['session_info']['time'],
                    'date': doc['session_info']['date']
                }
            }
            results.append(result)
        
        return results

def create_google_search_url(title: str) -> str:
    """Create a Google search URL for a given title"""
    encoded_title = urllib.parse.quote(title)
    return f"https://www.google.com/search?q={encoded_title}"

def main():
    st.set_page_config(
        page_title="Poster Search",
        page_icon="🔍",
        layout="wide"
    )
    
    # Minimal styling without background colors
    st.markdown("""
        <style>
        .stMarkdown {
            font-size: 1.2rem;
        }
        .session-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .session-item {
            margin: 0.5rem 0;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main title and description
    st.title("🔍 NeurIPS 2024 Poster Search")
    st.markdown("Search through conference posters using natural language queries.")
    
    # Load embeddings
    @st.cache_resource
    def load_embeddings(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        embedding_data = load_embeddings("embeddings/poster_embeddings.pkl")
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return
    
    # Search interface with better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your search query:", placeholder="e.g., machine learning in healthcare")
    
    with col2:
        num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    # Initialize processor with environment API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return
        
    processor = EmbeddingProcessor(api_key=api_key)
    
    if query:
        try:
            with st.spinner("Searching..."):
                results = processor.search_similar(query, embedding_data, num_results)
            
            st.header(f"🎯 Top {len(results)} Results")
            
            for i, result in enumerate(results, 1):
                google_url = create_google_search_url(result['title'])
                
                st.markdown("---")
                
                st.markdown(f"### [{result['title']}]({google_url})")
                st.markdown(f"*Similarity Score: {result['similarity']:.3f}*")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Poster Number:** {result['poster_number']}")
                    st.markdown(f"**Authors:** {result['authors']}")
                    st.markdown("**Abstract:**")
                    st.markdown(result['abstract'])
                
                with col2:
                    session = result['session_info']
                    st.markdown("### 📍 Session Information")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**🏷️ Session:** {session['session_name']}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**📍 Location:** {session['location']}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**🕒 Time:** {session['time']}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**📅 Date:** {session['date']}")
        
        except Exception as e:
            st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()