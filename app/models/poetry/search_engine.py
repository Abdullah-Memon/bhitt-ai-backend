import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
from collections import defaultdict, OrderedDict
import math
import time
import random
import json
import hashlib
from datetime import datetime, timedelta
from langchain_experimental.agents import create_csv_agent
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path

# Add Config class for API key
class Config:
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')

class SindhiPoetrySearchEngine:
    def __init__(self, csv_file_path=None, cache_size=100, cache_expiry_hours=24):
        """
        Initialize the Sindhi Poetry Search Engine with caching and memory
        """
        # Handle CSV file path
        if csv_file_path is None:
            current_dir = Path(__file__).parent
            csv_file_path = current_dir / "data" / "poetry_data.csv"
        
        self.csv_file_path = csv_file_path
        self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load and validate data
        try:
            self.df = pd.read_csv(csv_file_path)
            self.df = self.df.fillna('')  # Handle NaN values
            print(f"Successfully loaded {len(self.df)} poetry entries")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
        
        # Cache settings
        self.cache_size = cache_size
        self.cache_expiry_hours = cache_expiry_hours
        self.search_cache = OrderedDict()
        self.current_search_memory = {}
        
        # Initialize search components
        self._preprocess_data()
        self._setup_similarity_search()
        self._setup_bm25_search()

    def _remove_diacritics(self, text):
        """Remove diacritical marks from Arabic/Sindhi text"""
        if not text or pd.isna(text):
            return ''
        
        text = str(text)  # Ensure it's a string
        diacritics = [
            '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
            '\u0651', '\u0652', '\u0653', '\u0654', '\u0655', '\u0656',
            '\u0657', '\u0658', '\u0659', '\u065A', '\u065B', '\u065C',
            '\u065D', '\u065E', '\u065F', '\u0670', '\u0640'
        ]
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        return text

    def _normalize_text(self, text):
        """Normalize text for better matching: remove diacritics, extra spaces."""
        if not text or pd.isna(text):
            return ''
        
        text = str(text)  # Ensure it's a string
        text = self._remove_diacritics(text)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        return text.strip().lower()  # Convert to lowercase for better matching

    def _preprocess_data(self):
        """Preprocess the dataset: Normalize text fields once during loading."""
        print("Preprocessing data...")
        
        # Normalize key text fields
        for col in ['sur', 'compiler', 'keywords']:
            if col in self.df.columns:
                self.df[f'{col}_normalized'] = self.df[col].apply(self._normalize_text)
                print(f"Normalized column: {col}")
        
        # Ensure all text columns are strings
        text_columns = ['sur', 'text', 'dastan', 'dastan_verse_number', 
                       'poetry_text', 'explanation', 'compiler', 'keywords']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).fillna('')
        
        print("Data preprocessing complete")

    def _setup_similarity_search(self):
        """Setup TF-IDF vectorizer for semantic similarity search"""
        print("Setting up similarity search...")
        
        try:
            # Create combined text for similarity search
            combined_text = []
            for idx, row in self.df.iterrows():
                # Combine multiple fields for better search
                fields_to_combine = []
                
                # Add normalized fields if they exist
                for col in ['sur_normalized', 'compiler_normalized']:
                    if col in self.df.columns and row[col]:
                        fields_to_combine.append(str(row[col]))
                
                # Add original text fields as fallback
                for col in ['text', 'poetry_text']:
                    if col in self.df.columns and row[col]:
                        normalized_field = self._normalize_text(row[col])
                        if normalized_field:
                            fields_to_combine.append(normalized_field)
                
                combined = ' '.join(fields_to_combine).strip()
                combined_text.append(combined if combined else ' ')  # Ensure no empty strings
            
            # Setup TF-IDF vectorizer with improved settings
            self.poetry_vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=1,
                max_features=5000,
                stop_words=None,  # Don't remove stop words for non-English text
                lowercase=False,   # We already handle normalization
                token_pattern=r'[^\s]+',  # Better token pattern for Arabic/Sindhi
                norm='l2'
            )
            
            # Fit and transform the combined text
            self.poetry_tfidf_matrix = self.poetry_vectorizer.fit_transform(combined_text)
            
            print(f"TF-IDF matrix shape: {self.poetry_tfidf_matrix.shape}")
            print(f"Vocabulary size: {len(self.poetry_vectorizer.vocabulary_)}")
            print("Similarity search setup complete")
            
        except Exception as e:
            print(f"Error setting up similarity search: {e}")
            # Create a fallback empty vectorizer
            self.poetry_vectorizer = TfidfVectorizer()
            self.poetry_tfidf_matrix = None

    def _setup_bm25_search(self):
        """Setup BM25 search for keywords"""
        print("Setting up BM25 search...")
        
        try:
            self.k1 = 1.5
            self.b = 0.75
            
            # Process keywords for BM25
            if 'keywords_normalized' in self.df.columns:
                self.keyword_tokens = self.df['keywords_normalized'].apply(
                    lambda x: str(x).split() if x and not pd.isna(x) else []
                ).tolist()
            else:
                # Fallback to using other text fields
                self.keyword_tokens = []
                for idx, row in self.df.iterrows():
                    tokens = []
                    for col in ['sur', 'text', 'compiler']:
                        if col in self.df.columns and row[col]:
                            normalized = self._normalize_text(row[col])
                            if normalized:
                                tokens.extend(normalized.split())
                    self.keyword_tokens.append(tokens)
            
            # Calculate document frequencies
            self.doc_freqs = defaultdict(int)
            vocab = set()
            total_tokens = 0
            doc_count = 0
            
            for tokens in self.keyword_tokens:
                if tokens:  # Only process non-empty token lists
                    vocab.update(tokens)
                    total_tokens += len(tokens)
                    doc_count += 1
                    for token in set(tokens):
                        self.doc_freqs[token] += 1
            
            self.vocab = vocab
            self.avgdl = total_tokens / doc_count if doc_count > 0 else 0
            
            print(f"BM25 vocabulary size: {len(self.vocab)}")
            print(f"Average document length: {self.avgdl:.2f}")
            print("BM25 search setup complete")
            
        except Exception as e:
            print(f"Error setting up BM25 search: {e}")
            self.keyword_tokens = []
            self.doc_freqs = defaultdict(int)
            self.vocab = set()
            self.avgdl = 0

    def _get_cache_key(self, query, top_k, similarity_weight, bm25_weight):
        """Generate a cache key for the search parameters"""
        cache_data = f"{query}_{top_k}_{similarity_weight}_{bm25_weight}"
        return hashlib.md5(cache_data.encode('utf-8')).hexdigest()

    def _is_cache_expired(self, timestamp):
        """Check if cache entry is expired"""
        if not timestamp:
            return True
        cache_time = datetime.fromisoformat(timestamp)
        expiry_time = cache_time + timedelta(hours=self.cache_expiry_hours)
        return datetime.now() > expiry_time

    def _get_from_cache(self, cache_key):
        """Get results from cache if valid and not expired"""
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            if not self._is_cache_expired(cached_data.get('timestamp')):
                # Move to end (LRU)
                self.search_cache.move_to_end(cache_key)
                return cached_data['results']
            else:
                # Remove expired entry
                del self.search_cache[cache_key]
        return None

    def _save_to_cache(self, cache_key, results):
        """Save results to cache with timestamp"""
        # Remove oldest entries if cache is full
        while len(self.search_cache) >= self.cache_size:
            self.search_cache.popitem(last=False)
        
        self.search_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def search(self, query, top_k=10, similarity_weight=0.3, bm25_weight=0.7, use_cache=False):
        """
        Perform hybrid search combining similarity and BM25 with caching
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            similarity_weight (float): Weight for similarity scores
            bm25_weight (float): Weight for BM25 scores  
            use_cache (bool): Whether to use caching
        
        Returns:
            list: List of search results
        """
        if not query or not query.strip():
            print("Empty query provided")
            return []
        
        print(f"Searching for: '{query}' with parameters:")
        print(f"  top_k: {top_k}")
        print(f"  similarity_weight: {similarity_weight}")
        print(f"  bm25_weight: {bm25_weight}")
        print(f"  use_cache: {use_cache}")
        
        # Check cache first if enabled
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(query, top_k, similarity_weight, bm25_weight)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                print("Returning cached results")
                return cached_results
        
        try:
            # Normalize query
            normalized_query = self._normalize_text(query)
            print(f"Normalized query: '{normalized_query}'")
            
            # Initialize scores
            similarity_scores = np.zeros(len(self.df))
            bm25_scores = np.zeros(len(self.df))
            
            # Calculate similarity scores
            if similarity_weight > 0 and self.poetry_tfidf_matrix is not None:
                try:
                    query_vector = self.poetry_vectorizer.transform([normalized_query])
                    similarity_scores = cosine_similarity(query_vector, self.poetry_tfidf_matrix).flatten()
                    print(f"Similarity scores calculated. Max: {similarity_scores.max():.3f}")
                except Exception as e:
                    print(f"Error calculating similarity scores: {e}")
                    similarity_scores = np.zeros(len(self.df))
            
            # Calculate BM25 scores
            if bm25_weight > 0:
                try:
                    query_tokens = normalized_query.split()
                    bm25_scores = np.array([
                        self._bm25_score(query_tokens, doc_tokens) 
                        for doc_tokens in self.keyword_tokens
                    ])
                    print(f"BM25 scores calculated. Max: {bm25_scores.max():.3f}")
                except Exception as e:
                    print(f"Error calculating BM25 scores: {e}")
                    bm25_scores = np.zeros(len(self.df))
            
            # Normalize scores
            max_sim_score = similarity_scores.max() if similarity_scores.max() > 0 else 1
            max_bm25_score = bm25_scores.max() if bm25_scores.max() > 0 else 1
            
            if max_sim_score > 0:
                similarity_scores = similarity_scores / max_sim_score
            if max_bm25_score > 0:
                bm25_scores = bm25_scores / max_bm25_score
            
            # Combine scores
            combined_scores = similarity_weight * similarity_scores + bm25_weight * bm25_scores
            
            # Get top results
            all_indices = np.argsort(combined_scores)[::-1]
            all_results = []
            for idx in all_indices:
                if combined_scores[idx] > 0:
                    result = {
                        'index': int(idx),
                        'combined_score': float(combined_scores[idx]),
                        'similarity_score': float(similarity_scores[idx]),
                        'bm25_score': float(bm25_scores[idx]),
                        'sur': self.df.iloc[idx]['sur'],
                        'text': self.df.iloc[idx]['text'],
                        'dastan': self.df.iloc[idx]['dastan'],
                        'dastan_verse_number': self.df.iloc[idx]['dastan_verse_number'],
                        'poetry_text': self.df.iloc[idx]['poetry_text'],
                        'explanation': self.df.iloc[idx]['explanation'],
                        'compiler': self.df.iloc[idx]['compiler']
                    }
                    all_results.append(result)
            
            # Get top_k results
            results = all_results[:top_k]
            
            # Save to cache if enabled
            if use_cache and cache_key:
                self._save_to_cache(cache_key, results)
            
            print(f"Search completed, found {len(all_results)} results, returning top {len(results)}")
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []

    def _bm25_score(self, query_tokens, doc_tokens):
        """Calculate BM25 score for a document"""
        if not doc_tokens:  # Handle empty token lists
            return 0.0
            
        score = 0.0
        doc_len = len(doc_tokens)
        doc_len_factor = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl > 0 else self.k1
        
        for token in query_tokens:
            if token in self.doc_freqs and token in doc_tokens:
                tf = doc_tokens.count(token)
                df = self.doc_freqs[token]
                if df > 0:
                    idf = math.log((len(self.df) - df + 0.5) / (df + 0.5))
                    score += idf * (tf * (self.k1 + 1)) / (tf + doc_len_factor)
        return score

    def debug_search(self, query):
        """Debug method to help troubleshoot search issues"""
        print(f"\n=== DEBUG SEARCH FOR: '{query}' ===")
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Available columns: {list(self.df.columns)}")
        
        # Check TF-IDF setup
        if self.poetry_tfidf_matrix is not None:
            print(f"TF-IDF matrix shape: {self.poetry_tfidf_matrix.shape}")
            print(f"Vocabulary size: {len(self.poetry_vectorizer.vocabulary_)}")
        else:
            print("TF-IDF matrix is None!")
        
        # Check BM25 setup
        print(f"BM25 vocabulary size: {len(self.vocab)}")
        print(f"Average document length: {self.avgdl}")
        print(f"Number of keyword token lists: {len(self.keyword_tokens)}")
        
        # Test normalization
        normalized_query = self._normalize_text(query)
        print(f"Normalized query: '{normalized_query}'")
        
        # Test search with different parameters
        print("\nTesting different parameter combinations:")
        test_configs = [
            (1.0, 0.0, "Similarity only"),
            (0.0, 1.0, "BM25 only"),
            (0.7, 0.3, "Hybrid"),
            (0.5, 0.5, "Balanced")
        ]
        
        for sim_weight, bm25_weight, desc in test_configs:
            results = self.search(query, top_k=10, similarity_weight=sim_weight, bm25_weight=bm25_weight, use_cache=False)
            print(f"  {desc} ({sim_weight:.1f}, {bm25_weight:.1f}): {len(results)} results")
            if results:
                print(f"    Top score: {results[0]['combined_score']:.3f}")


# Agent code for result selection
def agent_select_best_result(query, search_results):
    """Use the agent to determine the most accurate result based on the user's query"""
    try:
        load_dotenv()
        chat = ChatTogether(
            together_api_key=Config.TOGETHER_API_KEY,
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0.2
        )
        # Format search results for the prompt
        results_text = ""
        for i, result in enumerate(search_results):
            results_text += f"""
Result {i+1}:
- Sur: {result['sur']}
- Text: {result['text']}
- Dastan: {result['dastan']}
- Poetry Text: {result['poetry_text']}
- Explanation: {result['explanation']}
- Combined Score: {result['combined_score']:.4f}
---"""
        # Prepare the prompt
        prompt = f"""
You are a Sindhi poetry expert. Based on the user's query: "{query}", analyze the following search results and select the TOP 3 MOST ACCURATE and RELEVANT results that best match the query.

Search Results:
{results_text}

Instructions:
1. Analyze each result carefully
2. Consider the relevance to the query
3. Return exactly 3 numbers (e.g., 1, 3, 7) separated by commas, representing the best matching results in order of preference
4. Do not provide explanations, just the three numbers
5. If fewer than 3 good results exist, still provide 3 numbers choosing the best available

Best result numbers:"""
        
        # Get response from the agent
        response = chat.invoke(prompt)
        
        # Extract the response content
        if isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Extract the numbers from response
        import re
        numbers = re.findall(r'\d+', response_text.strip())
        
        selected_results = []
        
        if numbers and len(numbers) >= 3:
            # Get the first 3 numbers and convert to 0-based indices
            for i in range(3):
                selected_index = int(numbers[i]) - 1  # Convert to 0-based index
                if 0 <= selected_index < len(search_results):
                    selected_results.append(search_results[selected_index])
        
        # If we don't have 3 results, fill with the top results
        while len(selected_results) < 3 and len(selected_results) < len(search_results):
            # Add results that weren't already selected
            for i, result in enumerate(search_results):
                if result not in selected_results:
                    selected_results.append(result)
                    break
        
        # Return the selected results (up to 3)
        return selected_results[:3] if selected_results else search_results[:3]
        
    except Exception as e:
        print(f"Error in agent selection: {e}")
        # Fallback: return the top 3 results
        return search_results[:3] if len(search_results) >= 3 else search_results

# Utility function for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Global search engine instance for optimization
_search_engine_instance = None

def get_search_engine_instance():
    """Get or create a singleton instance of the search engine for optimization"""
    global _search_engine_instance
    if _search_engine_instance is None:
        try:
            _search_engine_instance = SindhiPoetrySearchEngine()
            print("Search engine initialized successfully")
        except Exception as e:
            print(f"Error initializing search engine: {e}")
            raise
    return _search_engine_instance


def query_sindhi_poetry(query: str, show_best_only: bool = True) -> str:
    """
    Main function to query Sindhi poetry
    Args:
        query: Search query string
        show_best_only: If True, show only the agent-selected best result
    Returns:
        Formatted response string
    """
    try:
        # Use singleton instance for better performance
        search_engine = get_search_engine_instance()
        # Perform search
        results = search_engine.search(query, top_k=10, use_cache=False)
        if not results:
            return "معذرت، آپ کے سوال کا جواب نہیں ملا۔ براہ کرم دوسرے الفاظ میں سوال کریں۔"
        # If show_best_only is True, use agent to select best result
        if show_best_only:
            best_results = agent_select_best_result(query, results)
            if best_results:
                results = best_results  # Show all best results
        # Format response
        response_parts = []
        for i, result in enumerate(results, 1):
            # if show_best_only:
            #     response_parts.append(f"\n--- بہترین نتیجہ {i} (Agent Selected) ---")
            # else:
            response_parts.append(f"\n--- Result {i} ---")
            # Define the order of fields to display exactly like notebook
            field_order = ['sur', 'text', 'dastan', 'dastan_verse_number', 
                          'poetry_text', 'explanation', 'compiler']
            for field_name in field_order:
                if field_name in result and field_name not in ['index', 'combined_score', 'similarity_score', 'bm25_score']:
                    field_value = result[field_name]    
                    if field_value and str(field_value).strip():  # Only show non-empty fields
                        if field_name == 'poetry_text':
                            response_parts.append(f"Poetry Text: {field_value}")
                        elif field_name == 'sur':
                            response_parts.append(f"Sur: {field_value}")
                        elif field_name == 'text':
                            response_parts.append(f"Text: {field_value}")
                        elif field_name == 'dastan':
                            response_parts.append(f"Dastan: {field_value}")
                        elif field_name == 'dastan_verse_number':
                            response_parts.append(f"Dastan Verse Number: {field_value}")
                        elif field_name == 'explanation':
                            response_parts.append(f"Explanation: {field_value}")
                        elif field_name == 'compiler':
                            response_parts.append(f"Compiler: {field_value}")
            # Add separator between results
            response_parts.append("-" * 50)
        return "\n".join(response_parts)
    except Exception as e:
        import logging
        logging.error(f"Error in poetry search: {str(e)}")
        return f"خرابی ہوئی: {str(e)}"

def query_sindhi_poetry_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    """
    Session-aware function to query Sindhi poetry
    
    Args:
        query: Search query string
        user_id: ID of the user making the query
        session_id: Optional session ID. If None, a new session will be created
        
    Returns:
        dict: Response with session information
    """
    import logging
    
    try:
        # Import session service (adjust import path as needed)
        try:
            from ...services.session_service import SessionService
        except ImportError:
            logging.error("Could not import SessionService")
            return {
                'error': 'Session service not available',
                'code': 'SERVICE_UNAVAILABLE'
            }
        
        # Validate and handle session
        if session_id:
            # Verify session exists and belongs to user
            session = SessionService.get_session(session_id)
            if not session:
                return {
                    'error': 'Session not found',
                    'code': 'SESSION_NOT_FOUND'
                }
            
            if not SessionService.verify_session_belongs_to_user(session_id, user_id):
                return {
                    'error': 'Session does not belong to user',
                    'code': 'SESSION_UNAUTHORIZED'
                }
            
            # Use existing session (could be from chatbot or poetry)
            logging.info(f"Using existing session {session_id} for poetry search")
        else:
            # Create new session with poetry-specific name only if no session provided
            session_name = f"شاعری تلاش: {query[:80]}" if len(query) <= 80 else f"شاعری تلاش: {query[:77]}..."
            session_id = SessionService.create_session(user_id, session_name)
            logging.info(f"Created new poetry session {session_id}")
        
        # Save user message with poetry query
        user_message_id = SessionService.save_message(session_id, 'user', query)
        
        # Update session activity
        SessionService.update_session_activity(session_id)


        remove_words_list = [
    "جو", "جي", "۽", "کان", "کي", "هن", "ان", "ٻڌايو", "توهان", "اھا", 
    "ڪجھ", "اڳ", "ٿي", "ساٿ", "تي", "هڪ", "انهي", "تمام", "جيتوڻ", "ٿو", 
    "کڻي", "مونکي", "اڳ", "واسطو", "ڪمپني", "ٻي", "جڏهن", "اچو", "آڻڻ",
    "سڀ", "جيڪڏهن", "نه", "آهي", "ھڪ", "ساڳي","سر","ٻڌا",
]
        
        # loop to remove words from the query based on remove_words_list
        
        if remove_words_list:
            for word in remove_words_list:
                query = query.replace(word, '')
        query = query.strip()  # Clean up the query
        if not query:
            return {
                'error': 'invalid query',
                'code': 'INVALID_QUERY'
            }
        logging.info(f"Processing poetry search query: {query} in session {session_id}")
        
        # Get poetry search response
        response = query_sindhi_poetry(query)
        
        # Save bot response
        bot_message_id = SessionService.save_message(session_id, 'bot', response)
        
        # Ensure all data is JSON serializable
        response_data = {
            'query': query,
            'response': response,
            'session_id': session_id,
            'user_message_id': user_message_id,
            'bot_message_id': bot_message_id,
            'model': 'poetry-search-engine',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return convert_numpy_types(response_data)
        
    except Exception as e:
        logging.error(f"Error in session-aware poetry search: {str(e)}")
        return {
            'error': f"خرابی ہوئی: {str(e)}",
            'code': 'INTERNAL_ERROR',
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
