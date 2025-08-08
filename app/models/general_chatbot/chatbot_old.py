import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Disable tokenizer parallelism to avoid forking issues in production
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer, AutoModel
import torch
from together import Together
from langchain.prompts import PromptTemplate
import faiss
import numpy as np
from huggingface_hub import login
import hashlib
import pickle
from datetime import datetime, timedelta
from collections import OrderedDict, deque
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path
from ...config.config import Config

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ========== CONFIGURATION ==========

TOGETHER_API_KEY = getattr(Config, 'TOGETHER_API_KEY', '')
MODEL_NAME = getattr(Config, 'CHATBOT_MODEL_NAME', "BAAI/bge-m3")
DATA_PATH_CONFIG = getattr(Config, 'CHATBOT_DATA_PATH', 'app/models/general_chatbot/data/bhit_data.txt')

# Convert relative path to absolute
if not os.path.isabs(DATA_PATH_CONFIG):
    base_dir = Path(__file__).parent.parent.parent.parent
    DATA_PATH = base_dir / DATA_PATH_CONFIG
else:
    DATA_PATH = Path(DATA_PATH_CONFIG)

# Enhanced retrieval parameters - Reduced for production stability
INITIAL_RETRIEVE_K = getattr(Config, 'CHATBOT_INITIAL_RETRIEVE_K', 15)  # Reduced from 30
RERANK_K = getattr(Config, 'CHATBOT_RERANK_K', 8)  # Reduced from 15
FINAL_CONTEXT_K = getattr(Config, 'CHATBOT_FINAL_CONTEXT_K', 4)  # Reduced from 6
MIN_SIMILARITY_THRESHOLD = getattr(Config, 'CHATBOT_MIN_SIMILARITY_THRESHOLD', 0.1)
CONTEXT_EXPANSION_RADIUS = getattr(Config, 'CHATBOT_CONTEXT_EXPANSION_RADIUS', 1)  # Reduced from 2

# Conversational memory
MAX_CONVERSATION_HISTORY = getattr(Config, 'CHATBOT_MAX_CONVERSATION_HISTORY', 10)
CONVERSATION_CONTEXT_WEIGHT = 0.3

# Cache settings
CACHE_DIR = Path(__file__).parent / "cache"
RESPONSE_CACHE_SIZE = getattr(Config, 'CHATBOT_CACHE_SIZE', 2000)

# ========== DATA STRUCTURES ==========

@dataclass
class ConversationTurn:
    turn_id: str
    query: str
    answer: str
    context_chunks: List[str]
    timestamp: datetime
    confidence: float
    chunk_indices: List[int]

@dataclass
class QueryContext:
    original_query: str
    processed_query: str
    query_type: str
    expanded_terms: List[str]
    conversation_context: str
    related_history: List[ConversationTurn]

# ========== ENHANCED CACHING ==========

class EnhancedCache:
    def __init__(self, cache_dir: str, max_size: int = 2000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = OrderedDict()
        self.access_count = {}
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, data: str) -> str:
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[any]:
        if key in self.memory_cache:
            self.memory_cache.move_to_end(key)
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.memory_cache[key]
        
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if datetime.now() - cached_data['timestamp'] < timedelta(hours=48):
                    self._add_to_memory(key, cached_data['data'])
                    return cached_data['data']
                else:
                    os.remove(cache_path)
            except Exception:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        return None
    
    def _add_to_memory(self, key: str, data: any):
        if len(self.memory_cache) >= self.max_size:
            # Remove least frequently used items
            lfu_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self.memory_cache.pop(lfu_key, None)
            self.access_count.pop(lfu_key, None)
        
        self.memory_cache[key] = data
        self.access_count[key] = 1
    
    def set(self, key: str, data: any):
        self._add_to_memory(key, data)
        
        cache_data = {'data': data, 'timestamp': datetime.now()}
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logging.warning(f"Disk cache failed: {e}")

# ========== CONVERSATION MEMORY ==========

class ConversationMemory:
    def __init__(self, max_history: int = MAX_CONVERSATION_HISTORY):
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = str(uuid.uuid4())
        
    def add_turn(self, query: str, answer: str, context_chunks: List[str], 
                 confidence: float, chunk_indices: List[int]):
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            timestamp=datetime.now(),
            confidence=confidence,
            chunk_indices=chunk_indices
        )
        self.conversation_history.append(turn)
    
    def get_relevant_history(self, current_query: str) -> List[ConversationTurn]:
        """Get conversation turns relevant to current query"""
        if not self.conversation_history:
            return []
        
        relevant_turns = []
        query_words = set(current_query.lower().split())
        
        for turn in reversed(self.conversation_history):
            # Check for word overlap
            turn_words = set(turn.query.lower().split())
            overlap = len(query_words.intersection(turn_words))
            
            # Include if significant overlap or recent high-confidence turn
            if (overlap >= 2 or 
                (turn.confidence > 0.8 and 
                 (datetime.now() - turn.timestamp).seconds < 300)):
                relevant_turns.append(turn)
        
        return relevant_turns[:3]  # Most recent 3 relevant turns
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for prompting"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for turn in list(self.conversation_history)[-3:]:  # Last 3 turns
            context_parts.append(f"پراڻو سوال: {turn.query}")
            context_parts.append(f"پراڻو جواب: {turn.answer[:200]}...")
        
        return "\n".join(context_parts)

# ========== ENHANCED CHUNKING ==========

class PrecisionChunker:
    def __init__(self):
        self.sindhi_sentence_pattern = r'[۔؟!]+' 
        self.paragraph_pattern = r'\n\s*\n'
        
    def create_overlapping_chunks(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        """Create overlapping chunks with reduced size for better performance"""
        sentences = re.split(self.sindhi_sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        sentence_indices = []
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            sentence_indices.append(i)
            
            if len(" ".join(current_chunk).split()) >= chunk_size:
                chunk_data = {
                    'text': " ".join(current_chunk),
                    'sentence_indices': sentence_indices,
                    'start_sentence': sentence_indices[0],
                    'end_sentence': sentence_indices[-1],
                }
                chunks.append(chunk_data)
                
                current_chunk = current_chunk[-overlap:]  
                sentence_indices = sentence_indices[-overlap:]

        if current_chunk:
            chunk_data = {
                'text': " ".join(current_chunk),
                'sentence_indices': sentence_indices,
                'start_sentence': sentence_indices[0],
                'end_sentence': sentence_indices[-1],
            }
            chunks.append(chunk_data)
        
        return chunks

# ========== ENHANCED RAG SYSTEM ==========

class ProductionRAGSystem:
    def __init__(self):
        # Initialize models with memory optimization
        huggingface_token = getattr(Config, 'HUGGINGFACE_TOKEN', '')
        if huggingface_token:
            login(token=huggingface_token)
        
        # Load models with memory optimization
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            model_max_length=256,  # Limit max length for memory efficiency
            use_fast=True  # Use fast tokenizer if available
        )
        
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use half precision if GPU available
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Move model to eval mode and optimize
        self.model.eval()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.client = Together(api_key=TOGETHER_API_KEY)
        print("Models loaded successfully")
        
        # Initialize components
        self.embedding_cache = EnhancedCache(f"{CACHE_DIR}/embeddings")
        self.response_cache = EnhancedCache(f"{CACHE_DIR}/responses", RESPONSE_CACHE_SIZE)
        self.conversation_memory = ConversationMemory()
        self.chunker = PrecisionChunker()
        
        # Data storage
        self.original_text = ""
        self.chunk_data = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        
        # Enhanced prompt template
        self.prompt = PromptTemplate.from_template("""
توھان شاھ عبداللطيف ڀٽائي جي زندگي ۽ ڪم بابت تفصيل سان ڄاڻ رکندا آھيو۔

{conversation_context}

ھدايتون:
- صرف ڄاڻايل حوالو (Context) مان درست معلومات استعمال ڪريو
- صرف سنڌي ٻولي ۾ جواب ڏيو
- اردو، انگريزي يا ٻين ٻولين جا الفاظ استعمال نه ڪريو
- جڏھن توھان کان اھو سوال پڇو وڃي ت شاھ لطيف جي پُٽ جو نالو ڇا ھو؟ ت توھان  کي جواب ڏيڻو آھي ت  شاھ لطيف ڪو ب پُٽ ن ھو
- صرف ڏنل حوالي (Context) مان مهرباني ڪري جواب ڏيو
موجوده سوال: {question}

حوالو:
{context}

اعتماد جو درجو: {confidence}/1.0

جواب:
""")
        
        # Load and process data
        self.load_and_process_data()
    
    def load_and_process_data(self, file_path: str = None):
        """Enhanced data loading and processing"""
        print("🔄 Loading and processing data...")
        
        # Use provided file_path or default DATA_PATH
        data_path = file_path if file_path else DATA_PATH
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.original_text = f.read()
        
        # Create precision chunks
        self.chunk_data = self.chunker.create_overlapping_chunks(self.original_text)
        print(f"✅ Created {len(self.chunk_data)} precision chunks")
        
        # Extract chunk texts for embedding
        chunk_texts = [chunk['text'] for chunk in self.chunk_data]
        
        # Generate embeddings
        print("🧠 Generating embeddings...")
        self.embeddings = self.generate_embeddings_batch(chunk_texts)
        
        # Create FAISS index
        print("🔍 Building search indices...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
        
        # Create BM25 index
        tokenized_chunks = [chunk['text'].split() for chunk in self.chunk_data]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        print("🚀 System ready for queries!")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Optimized batch embedding generation with safer processing and caching"""
        # First try to get cached embeddings
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self.embedding_cache._get_cache_key(text)
            cached_embedding = self.embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts using safe batch processing
        if uncached_texts:
            print(f"Generating embeddings for {len(uncached_texts)} new texts...")
            try:
                # Use smaller, safer batch processing - no multiprocessing in production
                batch_embeddings = self._generate_embeddings_safe_batch(uncached_texts)
                
                # Cache new embeddings
                for i, embedding in enumerate(batch_embeddings):
                    text_idx = uncached_indices[i]
                    cache_key = self.embedding_cache._get_cache_key(uncached_texts[i])
                    self.embedding_cache.set(cache_key, embedding)
                    embeddings.append((text_idx, embedding))
                    
            except Exception as e:
                logger.error(f"Error in embedding generation: {e}")
                # Final fallback to individual processing
                for i, text in enumerate(uncached_texts):
                    try:
                        text_idx = uncached_indices[i]
                        embedding = self._generate_single_embedding(text)
                        cache_key = self.embedding_cache._get_cache_key(text)
                        self.embedding_cache.set(cache_key, embedding)
                        embeddings.append((text_idx, embedding))
                    except Exception as e:
                        logger.error(f"Error generating single embedding: {e}")
                        # Create zero embedding as fallback
                        text_idx = uncached_indices[i]
                        zero_embedding = np.zeros(768)
                        embeddings.append((text_idx, zero_embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def _generate_embeddings_safe_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using safe batch processing without multiprocessing"""
        all_embeddings = []
        
        # Use very small batches to prevent memory issues and timeouts
        batch_size = 2  # Very conservative batch size for production
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Processing {len(texts)} texts in {total_batches} small batches...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                print(f"  Batch {batch_num}/{total_batches}: {len(batch_texts)} texts")
                
                # Tokenize with conservative settings
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=256  # Reduced for faster processing
                )
                
                # Generate embeddings with no gradient computation
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract embeddings and convert to numpy
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Clear memory after each batch
                del inputs, outputs, batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                print(f"  ✅ Batch {batch_num}/{total_batches} completed")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                # Generate zero embeddings for failed batch
                for _ in batch_texts:
                    all_embeddings.append(np.zeros(768).tolist())
                print(f"  ⚠️ Batch {batch_num} failed, using zero embeddings")
        
        return all_embeddings
    
    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text as fallback"""
        try:
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=256
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Clear memory
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return np.zeros(768)

    
    def expand_context_with_neighbors(self, selected_indices: List[int]) -> List[int]:
        """Expand context by including neighboring chunks"""
        expanded_indices = set(selected_indices)
        
        for idx in selected_indices:
            # Add neighboring chunks
            for offset in range(-CONTEXT_EXPANSION_RADIUS, CONTEXT_EXPANSION_RADIUS + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.chunk_data):
                    expanded_indices.add(neighbor_idx)
        
        return sorted(list(expanded_indices))
    
    def enhanced_query_processing(self, query: str) -> QueryContext:
        """Advanced query processing with conversation awareness"""
        # Get conversation context
        relevant_history = self.conversation_memory.get_relevant_history(query)
        conversation_context = self.conversation_memory.get_conversation_context()
        
        # Enhanced query expansion using conversation history
        expanded_terms = [query]
        for turn in relevant_history:
            # Extract key terms from relevant previous queries
            prev_words = turn.query.split()
            for word in prev_words:
                if len(word) > 3 and word not in expanded_terms:
                    expanded_terms.append(word)
        
        # Query type classification with conversation context
        query_type = self._classify_query_with_context(query, conversation_context)
        
        return QueryContext(
            original_query=query,
            processed_query=self._clean_query(query),
            query_type=query_type,
            expanded_terms=expanded_terms[:10],  # Limit expansion
            conversation_context=conversation_context,
            related_history=relevant_history
        )
    
    def _classify_query_with_context(self, query: str, context: str) -> str:
        """Enhanced query classification using conversation context"""
        combined_text = f"{context} {query}".lower()
        
        # Enhanced classification patterns
        if any(term in combined_text for term in ["ڪيڏانهن", "ڪٿي", "جنم", "ڄائو"]):
            return "birth_location"
        elif any(term in combined_text for term in ["ڪڏھن", "سال", "تاريخ", "ڄمڻ"]):
            return "birth_date"
        elif any(term in combined_text for term in ["شاعري", "ڪلام", "سُر", "رسالو", "شعر"]):
            return "poetry_work"
        elif any(term in combined_text for term in ["مرڻ", "وفات", "آخر", "موت"]):
            return "death"
        elif any(term in combined_text for term in ["زندگي", "حالات", "تعليم", "پيدائش"]):
            return "biography"
        elif any(term in combined_text for term in ["فلسفو", "تصوف", "عقيدو", "خيال"]):
            return "philosophy"
        else:
            return "general"
    
    def precision_retrieval(self, query_context: QueryContext) -> List[Tuple[Dict, float]]:
        """Enhanced retrieval with conversation awareness"""
        query = query_context.processed_query
        
        # Multi-query search using expanded terms
        all_results = {}
        
        # Primary query search
        query_embedding = self.generate_embeddings_batch([query])
        distances, indices = self.faiss_index.search(query_embedding, INITIAL_RETRIEVE_K * 2)
        
        # Score normalization and storage
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 / (1 + dist)
            if similarity > MIN_SIMILARITY_THRESHOLD:
                if idx not in all_results:
                    all_results[idx] = {'semantic': similarity, 'bm25': 0, 'conversation': 0}
                else:
                    all_results[idx]['semantic'] = max(all_results[idx]['semantic'], similarity)
        
        # BM25 search
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-INITIAL_RETRIEVE_K:]
        
        for idx in top_bm25_indices:
            if idx not in all_results:
                all_results[idx] = {'semantic': 0, 'bm25': bm25_scores[idx], 'conversation': 0}
            else:
                all_results[idx]['bm25'] = bm25_scores[idx]
        
        # Conversation context boost
        for turn in query_context.related_history:
            for chunk_idx in turn.chunk_indices:
                if chunk_idx in all_results:
                    all_results[chunk_idx]['conversation'] = turn.confidence * 0.3
        
        # Calculate hybrid scores
        results = []
        for idx, scores in all_results.items():
            hybrid_score = (0.5 * scores['semantic'] + 
                          0.3 * scores['bm25'] + 
                          0.2 * scores['conversation'])
            
            # Query type boosting
            chunk_text = self.chunk_data[idx]['text'].lower()
            if query_context.query_type == "birth_location" and any(term in chunk_text for term in ["ڀٽ شاھ", "ڄنم"]):
                hybrid_score *= 1.4
            elif query_context.query_type == "poetry_work" and any(term in chunk_text for term in ["رسالو", "سُر", "شاعري"]):
                hybrid_score *= 1.3
            elif query_context.query_type == "death" and any(term in chunk_text for term in ["وفات", "مرڻ", "انتقال"]):
                hybrid_score *= 1.4
            
            results.append((self.chunk_data[idx], hybrid_score))
        
        # Sort and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:RERANK_K]
    
    def verify_answer_accuracy(self, answer: str, context_chunks: List[str]) -> float:
        """Verify answer accuracy against source context"""
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for chunk in context_chunks:
            context_words.update(chunk.lower().split())
        
        # Calculate overlap ratio
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len([word for word in answer_words if len(word) > 2])
        
        if total_answer_words == 0:
            return 0.0
        
        accuracy_ratio = overlap / total_answer_words
        return min(1.0, accuracy_ratio * 1.2)  # Slight boost for good coverage
    
    def _clean_query(self, query: str) -> str:
        """Enhanced query cleaning"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        # Remove non-Sindhi characters except basic punctuation
        cleaned = re.sub(r'[^\u0600-\u06FF\s\?\.\!،]', '', cleaned)
        return cleaned
    
    def _expand_context_with_neighbors(self, selected_indices: List[int]) -> List[int]:
        """Expand context by including neighboring chunks"""
        expanded_indices = set(selected_indices)
        
        for idx in selected_indices:
            # Add neighboring chunks
            for offset in range(-CONTEXT_EXPANSION_RADIUS, CONTEXT_EXPANSION_RADIUS + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.chunk_data):
                    expanded_indices.add(neighbor_idx)
        
        return sorted(list(expanded_indices))
    
    def _calculate_enhanced_confidence(self, context_chunks: List[str], query: str, 
                                     query_context: QueryContext) -> float:
        """Enhanced confidence calculation"""
        if not context_chunks:
            return 0.0
        
        # Base semantic similarity
        chunk_embeddings = self.generate_embeddings_batch(context_chunks)
        query_embedding = self.generate_embeddings_batch([query])
        
        similarities = []
        for chunk_emb in chunk_embeddings:
            sim = np.dot(query_embedding[0], chunk_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_emb)
            )
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # Conversation context boost
        context_boost = 0.1 if len(query_context.related_history) > 0 else 0.0
        
        # Query type specific boost
        type_boost = 0.1 if query_context.query_type != "general" else 0.0
        
        # Final confidence - convert numpy type to Python float
        confidence = float(avg_similarity) + context_boost + type_boost
        return max(0.0, min(1.0, confidence))
    
    def _generate_fallback_response(self, context_text: str, query: str, query_context: QueryContext) -> str:
        """Generate a fallback response when API is not available"""
        if not context_text.strip():
            return "معذرت، سوال جو جواب موجود ڊيٽا ۾ نه مليو آهي۔"
        
        # Simple keyword-based response generation using the context
        query_lower = query.lower()
        context_lower = context_text.lower()
        
        # Extract relevant sentences from context
        sentences = re.split(r'[۔؟!]+', context_text)
        relevant_sentences = []
        
        query_words = set(query.split())
        for sentence in sentences:
            if sentence.strip():
                sentence_words = set(sentence.split())
                # Check if sentence contains query words
                if len(query_words.intersection(sentence_words)) > 0:
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Return the most relevant sentences
            response = "۔ ".join(relevant_sentences[:3])  # First 3 relevant sentences
            if len(response) > 500:
                response = response[:500] + "..."
            return response + "۔"
        
        # If no specific match, return first part of context
        first_part = context_text[:300]
        if len(context_text) > 300:
            first_part += "..."
        return first_part
    
    def get_enhanced_response(self, query: str) -> Dict[str, any]:
        """Generate enhanced response with conversation memory"""
        try:
            # Check cache with conversation context
            cache_key = f"{query}_{len(self.conversation_memory.conversation_history)}"
            cache_key_hash = self.response_cache._get_cache_key(cache_key)
            cached_response = self.response_cache.get(cache_key_hash)
            
            if cached_response:
                return cached_response
            
            print(f"Processing query: {query}")
            
            # Process query with conversation context
            query_context = self.enhanced_query_processing(query)
            
            # Precision retrieval
            retrieval_results = self.precision_retrieval(query_context)
            
            if not retrieval_results:
                return {
                    'query': query,
                    'answer': "معذرت، سوال جا حوالي ڊيٽا ۾ نه مليا آهن۔",
                    'confidence': 0.0,
                    'accuracy_score': 0.0,
                    'context_chunks_used': 0,
                    'conversation_context_used': False,
                    'retrieval_method': 'no_results',
                    'chunk_indices': []
                }
            
            # Expand context with neighboring chunks
            selected_indices = []
            for chunk_info, _ in retrieval_results[:FINAL_CONTEXT_K]:
                for i, chunk in enumerate(self.chunk_data):
                    if chunk['text'] == chunk_info['text']:
                        selected_indices.append(i)
                        break
            
            expanded_indices = self.expand_context_with_neighbors(selected_indices)
            
            # Assemble enhanced context
            context_chunks = [self.chunk_data[i]['text'] for i in expanded_indices]
            context_text = "\n\n".join(context_chunks)
            
            # Calculate confidence
            confidence = self._calculate_enhanced_confidence(context_chunks, query, query_context)
            
            # Generate response
            formatted_prompt = self.prompt.format(
                question=query,
                context=context_text,
                confidence=f"{confidence:.2f}",
                conversation_context=query_context.conversation_context
            )
            
            print("Generating AI response...")
            try:
                # Use a simpler timeout approach that works across platforms
                response = self.client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[
                        {"role": "system", "content": "توھان شاھ عبداللطيف ڀٽائي جي معلومات جي باري ۾ ماھر آھيو۔ صرف ڄاڻايل حوالن مان معلومات ڏيو۔"},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.1,  # Lower temperature for accuracy
                    max_tokens=600,  # Further reduced for faster response
                    timeout=10  # 10 second timeout
                )
                
                answer = response.choices[0].message.content
                print("AI response generated successfully")
                
            except Exception as e:
                logger.error(f"Error calling Together API: {e}")
                print(f"API call failed, using fallback response: {str(e)[:100]}")
                # Use fallback response
                answer = self._generate_fallback_response(context_text, query, query_context)
            
            # Verify answer accuracy (get context chunks again for verification)
            context_chunks_for_verification = [self.chunk_data[i]['text'] for i in expanded_indices]
            accuracy_score = self.verify_answer_accuracy(answer, context_chunks_for_verification)
            final_confidence = min(confidence, accuracy_score)
            
            # Clear memory after processing
            del context_chunks_for_verification, context_text, formatted_prompt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Prepare result with numpy type conversion
            result = {
                'query': query,
                'answer': answer,
                'confidence': convert_numpy_types(final_confidence),
                'accuracy_score': convert_numpy_types(accuracy_score),
                'context_chunks_used': convert_numpy_types(len([self.chunk_data[i]['text'] for i in expanded_indices])),
                'conversation_context_used': len(query_context.related_history) > 0,
                'retrieval_method': 'hybrid_with_conversation_optimized',
                'chunk_indices': convert_numpy_types(expanded_indices)
            }
            
            # Add to conversation memory
            context_chunks_for_memory = [self.chunk_data[i]['text'] for i in expanded_indices]
            self.conversation_memory.add_turn(
                query=query,
                answer=answer,
                context_chunks=context_chunks_for_memory,
                confidence=final_confidence,
                chunk_indices=expanded_indices
            )
            
            # Cache result
            self.response_cache.set(cache_key_hash, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in get_enhanced_response: {e}")
            return {
                'query': query,
                'answer': f"معذرت، سسٽم ۾ خرابي آئي آهي: {str(e)[:100]}",
                'confidence': 0.0,
                'accuracy_score': 0.0,
                'context_chunks_used': 0,
                'conversation_context_used': False,
                'retrieval_method': 'error',
                'chunk_indices': []
            }
    
# Global instance
_rag_system = None

def get_rag_system():
    """Get or create the RAG system instance"""
    global _rag_system
    if _rag_system is None:
        _rag_system = ProductionRAGSystem()
    return _rag_system

def query_general_chatbot(query: str) -> str:
    """
    Main function to query the general chatbot
    
    Args:
        query: User query string
        
    Returns:
        Response string
    """
    try:
        rag_system = get_rag_system()
        result = rag_system.get_enhanced_response(query)
        return result['answer']
        
    except Exception as e:
        logger.error(f"Error in general chatbot: {str(e)}")
        return f"معذرت، خرابي آئي آهي: {str(e)}"

def query_general_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    """
    Session-aware function to query the general chatbot
    
    Args:
        query: User query string
        user_id: ID of the user making the query
        session_id: Optional session ID. If None, a new session will be created
        
    Returns:
        dict: Response with session information
    """
    from ...services.session_service import SessionService
    
    try:
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
        else:
            # Create new session with query as session name
            session_name = query[:100] if len(query) <= 100 else query[:97] + "..."
            session_id = SessionService.create_session(user_id, session_name)
        
        # Save user message
        user_message_id = SessionService.save_message(session_id, 'user', query)
        
        # Update session activity
        SessionService.update_session_activity(session_id)
        
        # Get chatbot response
        rag_system = get_rag_system()
        result = rag_system.get_enhanced_response(query)
        
        # Save bot response
        bot_message_id = SessionService.save_message(session_id, 'bot', result['answer'])
        
        # Convert numpy types to native Python types for JSON serialization
        response_data = {
            'query': query,
            'answer': result['answer'],
            'session_id': session_id,
            'user_message_id': user_message_id,
            'bot_message_id': bot_message_id,
            'confidence': convert_numpy_types(result.get('confidence', 0.0)),
            'accuracy_score': convert_numpy_types(result.get('accuracy_score', 0.0)),
            'context_chunks_used': convert_numpy_types(result.get('context_chunks_used', 0)),
            'model': 'general-chatbot'
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in session-aware general chatbot: {str(e)}")
        return {
            'error': f"معذرت، خرابي آئي آهي: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }
