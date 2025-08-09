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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    logger.debug(f"Converting numpy types for object: {type(obj)}")
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

# Enhanced retrieval parameters
INITIAL_RETRIEVE_K = getattr(Config, 'CHATBOT_INITIAL_RETRIEVE_K', 15)
RERANK_K = getattr(Config, 'CHATBOT_RERANK_K', 8)
FINAL_CONTEXT_K = getattr(Config, 'CHATBOT_FINAL_CONTEXT_K', 4)
MIN_SIMILARITY_THRESHOLD = getattr(Config, 'CHATBOT_MIN_SIMILARITY_THRESHOLD', 0.1)
CONTEXT_EXPANSION_RADIUS = getattr(Config, 'CHATBOT_CONTEXT_EXPANSION_RADIUS', 1)

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
        logger.info(f"Initializing EnhancedCache with directory {cache_dir} and max_size {max_size}")
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = OrderedDict()
        self.access_count = {}
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"Cache directory created/verified: {cache_dir}")
        
    def _get_cache_key(self, data: str) -> str:
        key = hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]
        logger.debug(f"Generated cache key: {key} for data length: {len(data)}")
        return key
    
    def get(self, key: str) -> Optional[any]:
        logger.debug(f"Checking cache for key: {key}")
        if key in self.memory_cache:
            self.memory_cache.move_to_end(key)
            self.access_count[key] = self.access_count.get(key, 0) + 1
            logger.debug(f"Cache hit for key: {key}")
            return self.memory_cache[key]
        
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if datetime.now() - cached_data['timestamp'] < timedelta(hours=48):
                    self._add_to_memory(key, cached_data['data'])
                    logger.debug(f"Disk cache hit for key: {key}")
                    return cached_data['data']
                else:
                    os.remove(cache_path)
                    logger.debug(f"Removed expired cache file: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to read cache file {cache_path}: {e}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def _add_to_memory(self, key: str, data: any):
        logger.debug(f"Adding to memory cache: {key}")
        if len(self.memory_cache) >= self.max_size:
            lfu_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self.memory_cache.pop(lfu_key, None)
            self.access_count.pop(lfu_key, None)
            logger.debug(f"Removed least frequently used cache item: {lfu_key}")
        
        self.memory_cache[key] = data
        self.access_count[key] = 1
    
    def set(self, key: str, data: any):
        logger.debug(f"Setting cache for key: {key}")
        self._add_to_memory(key, data)
        
        cache_data = {'data': data, 'timestamp': datetime.now()}
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cached to disk: {cache_path}")
        except Exception as e:
            logger.warning(f"Disk cache failed for {cache_path}: {e}")

# ========== CONVERSATION MEMORY ==========

class ConversationMemory:
    def __init__(self, max_history: int = MAX_CONVERSATION_HISTORY):
        logger.info(f"Initializing ConversationMemory with max_history: {max_history}")
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = str(uuid.uuid4())
        logger.debug(f"Created new session with ID: {self.session_id}")
        
    def add_turn(self, query: str, answer: str, context_chunks: List[str], 
                 confidence: float, chunk_indices: List[int]):
        logger.debug(f"Adding conversation turn for query: {query[:50]}...")
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
        logger.debug(f"Added turn with ID: {turn.turn_id}, confidence: {confidence}")
    
    def get_relevant_history(self, current_query: str) -> List[ConversationTurn]:
        logger.debug(f"Getting relevant history for query: {current_query[:50]}...")
        if not self.conversation_history:
            logger.debug("No conversation history available")
            return []
        
        relevant_turns = []
        query_words = set(current_query.lower().split())
        
        for turn in reversed(self.conversation_history):
            turn_words = set(turn.query.lower().split())
            overlap = len(query_words.intersection(turn_words))
            
            if (overlap >= 2 or 
                (turn.confidence > 0.8 and 
                 (datetime.now() - turn.timestamp).seconds < 300)):
                relevant_turns.append(turn)
                logger.debug(f"Found relevant turn: {turn.turn_id}, overlap: {overlap}")
        
        logger.debug(f"Returning {len(relevant_turns)} relevant turns")
        return relevant_turns[:3]
    
    def get_conversation_context(self) -> str:
        logger.debug("Generating conversation context")
        if not self.conversation_history:
            logger.debug("No conversation history for context")
            return ""
        
        context_parts = []
        for turn in list(self.conversation_history)[-3:]:
            context_parts.append(f"پراڻو سوال: {turn.query}")
            context_parts.append(f"پراڻو جواب: {turn.answer[:200]}...")
        
        context = "\n".join(context_parts)
        logger.debug(f"Generated conversation context with {len(context_parts)} parts")
        return context

# ========== ENHANCED CHUNKING ==========

class PrecisionChunker:
    def __init__(self):
        logger.info("Initializing PrecisionChunker")
        self.sindhi_sentence_pattern = r'[۔؟!]+' 
        self.paragraph_pattern = r'\n\s*\n'
        
    def create_overlapping_chunks(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        logger.info(f"Creating overlapping chunks with size {chunk_size} and overlap {overlap}")
        sentences = re.split(self.sindhi_sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug(f"Split text into {len(sentences)} sentences")
        
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
                logger.debug(f"Created chunk {len(chunks)} with {len(sentence_indices)} sentences")
                
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
            logger.debug(f"Created final chunk with {len(sentence_indices)} sentences")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

# ========== ENHANCED RAG SYSTEM ==========

class ProductionRAGSystem:
    def __init__(self):
        logger.info("Initializing ProductionRAGSystem")
        # Initialize models with memory optimization
        huggingface_token = getattr(Config, 'HUGGINGFACE_TOKEN', '')
        if huggingface_token:
            logger.debug("Logging into HuggingFace with provided token")
            login(token=huggingface_token)
        
        # Load models with memory optimization
        logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            model_max_length=256,
            use_fast=True
        )
        logger.debug("Tokenizer loaded successfully")
        
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.model.eval()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.debug("Enabled gradient checkpointing")
        
        self.client = Together(api_key=TOGETHER_API_KEY)
        logger.info("Together API client initialized")
        
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
        logger.debug("Setting up prompt template")
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
        logger.info(f"Loading and processing data from {file_path or DATA_PATH}")
        data_path = file_path if file_path else DATA_PATH
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.original_text = f.read()
            logger.debug(f"Loaded text data, length: {len(self.original_text)} characters")
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise
        
        # Create precision chunks
        self.chunk_data = self.chunker.create_overlapping_chunks(self.original_text)
        logger.info(f"Created {len(self.chunk_data)} precision chunks")
        
        # Extract chunk texts for embedding
        chunk_texts = [chunk['text'] for chunk in self.chunk_data]
        
        # Generate embeddings
        logger.info("Generating embeddings")
        self.embeddings = self.generate_embeddings_batch(chunk_texts)
        logger.debug(f"Generated embeddings shape: {self.embeddings.shape}")
        
        # Create FAISS index
        logger.info("Building FAISS index")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
        logger.debug(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Create BM25 index
        logger.info("Building BM25 index")
        tokenized_chunks = [chunk['text'].split() for chunk in self.chunk_data]
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.debug("BM25 index built")
        
        logger.info("System ready for queries")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self.embedding_cache._get_cache_key(text)
            cached_embedding = self.embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
                logger.debug(f"Using cached embedding for text {i}")
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached texts")
            try:
                batch_embeddings = self._generate_embeddings_safe_batch(uncached_texts)
                
                for i, embedding in enumerate(batch_embeddings):
                    text_idx = uncached_indices[i]
                    cache_key = self.embedding_cache._get_cache_key(uncached_texts[i])
                    self.embedding_cache.set(cache_key, embedding)
                    embeddings.append((text_idx, embedding))
                    logger.debug(f"Cached new embedding for text {text_idx}")
                    
            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}")
                for i, text in enumerate(uncached_texts):
                    try:
                        text_idx = uncached_indices[i]
                        embedding = self._generate_single_embedding(text)
                        cache_key = self.embedding_cache._get_cache_key(text)
                        self.embedding_cache.set(cache_key, embedding)
                        embeddings.append((text_idx, embedding))
                        logger.debug(f"Generated single embedding for text {text_idx}")
                    except Exception as e:
                        logger.error(f"Error generating single embedding: {e}")
                        text_idx = uncached_indices[i]
                        zero_embedding = np.zeros(768)
                        embeddings.append((text_idx, zero_embedding))
                        logger.debug(f"Using zero embedding for text {text_idx}")
        
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        logger.debug(f"Returning embeddings array with shape: {result.shape}")
        return result
    
    def _generate_embeddings_safe_batch(self, texts: List[str]) -> List[np.ndarray]:
        logger.info(f"Processing {len(texts)} texts in safe batch mode")
        all_embeddings = []
        batch_size = 2
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch_texts)} texts")
            
            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=256
                )
                logger.debug(f"Tokenized batch {batch_num}")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logger.debug(f"Generated model outputs for batch {batch_num}")
                
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(batch_embeddings.tolist())
                logger.debug(f"Processed batch {batch_num} successfully")
                
                del inputs, outputs, batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                for _ in batch_texts:
                    all_embeddings.append(np.zeros(768).tolist())
                logger.debug(f"Using zero embeddings for failed batch {batch_num}")
        
        logger.info(f"Completed batch embedding generation, total embeddings: {len(all_embeddings)}")
        return all_embeddings
    
    def _generate_single_embedding(self, text: str) -> np.ndarray:
        logger.debug(f"Generating single embedding for text: {text[:50]}...")
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
            logger.debug("Single embedding generated successfully")
            
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return np.zeros(768)

    def expand_context_with_neighbors(self, selected_indices: List[int]) -> List[int]:
        logger.debug(f"Expanding context for indices: {selected_indices}")
        expanded_indices = set(selected_indices)
        
        for idx in selected_indices:
            for offset in range(-CONTEXT_EXPANSION_RADIUS, CONTEXT_EXPANSION_RADIUS + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.chunk_data):
                    expanded_indices.add(neighbor_idx)
        
        result = sorted(list(expanded_indices))
        logger.debug(f"Expanded to indices: {result}")
        return result
    
    def enhanced_query_processing(self, query: str) -> QueryContext:
        logger.info(f"Processing query: {query}")
        relevant_history = self.conversation_memory.get_relevant_history(query)
        conversation_context = self.conversation_memory.get_conversation_context()
        logger.debug(f"Retrieved {len(relevant_history)} relevant history turns")
        
        expanded_terms = [query]
        for turn in relevant_history:
            prev_words = turn.query.split()
            for word in prev_words:
                if len(word) > 3 and word not in expanded_terms:
                    expanded_terms.append(word)
        
        query_type = self._classify_query_with_context(query, conversation_context)
        logger.debug(f"Classified query as type: {query_type}")
        
        result = QueryContext(
            original_query=query,
            processed_query=self._clean_query(query),
            query_type=query_type,
            expanded_terms=expanded_terms[:10],
            conversation_context=conversation_context,
            related_history=relevant_history
        )
        logger.debug(f"Created QueryContext with {len(expanded_terms)} expanded terms")
        return result
    
    def _classify_query_with_context(self, query: str, context: str) -> str:
        logger.debug(f"Classifying query with context: {query[:50]}...")
        combined_text = f"{context} {query}".lower()
        
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
        logger.info(f"Performing precision retrieval for query: {query_context.original_query[:50]}...")
        query = query_context.processed_query
        all_results = {}
        
        # Semantic search
        logger.debug("Generating query embedding for semantic search")
        query_embedding = self.generate_embeddings_batch([query])
        distances, indices = self.faiss_index.search(query_embedding, INITIAL_RETRIEVE_K * 2)
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 / (1 + dist)
            if similarity > MIN_SIMILARITY_THRESHOLD:
                all_results[idx] = {'semantic': similarity, 'bm25': 0, 'conversation': 0}
                logger.debug(f"Added semantic result for chunk {idx}: similarity={similarity}")
        
        # BM25 search
        logger.debug("Performing BM25 search")
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-INITIAL_RETRIEVE_K:]
        
        for idx in top_bm25_indices:
            if idx not in all_results:
                all_results[idx] = {'semantic': 0, 'bm25': bm25_scores[idx], 'conversation': 0}
            else:
                all_results[idx]['bm25'] = bm25_scores[idx]
            logger.debug(f"Added BM25 result for chunk {idx}: score={bm25_scores[idx]}")
        
        # Conversation context boost
        for turn in query_context.related_history:
            for chunk_idx in turn.chunk_indices:
                if chunk_idx in all_results:
                    all_results[chunk_idx]['conversation'] = turn.confidence * 0.3
                    logger.debug(f"Boosted chunk {chunk_idx} with conversation score: {turn.confidence * 0.3}")
        
        # Calculate hybrid scores
        results = []
        for idx, scores in all_results.items():
            hybrid_score = (0.5 * scores['semantic'] + 
                          0.3 * scores['bm25'] + 
                          0.2 * scores['conversation'])
            
            chunk_text = self.chunk_data[idx]['text'].lower()
            if query_context.query_type == "birth_location" and any(term in chunk_text for term in ["ڀٽ شاھ", "ڄنم"]):
                hybrid_score *= 1.4
                logger.debug(f"Boosted chunk {idx} for birth_location")
            elif query_context.query_type == "poetry_work" and any(term in chunk_text for term in ["رسالو", "سُر", "شاعري"]):
                hybrid_score *= 1.3
                logger.debug(f"Boosted chunk {idx} for poetry_work")
            elif query_context.query_type == "death" and any(term in chunk_text for term in ["وفات", "مرڻ", "انتقال"]):
                hybrid_score *= 1.4
                logger.debug(f"Boosted chunk {idx} for death")
            
            results.append((self.chunk_data[idx], hybrid_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Retrieved {len(results[:RERANK_K])} top results")
        return results[:RERANK_K]
    
    def verify_answer_accuracy(self, answer: str, context_chunks: List[str]) -> float:
        logger.debug(f"Verifying answer accuracy for answer length: {len(answer)}")
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for chunk in context_chunks:
            context_words.update(chunk.lower().split())
        
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len([word for word in answer_words if len(word) > 2])
        
        if total_answer_words == 0:
            logger.debug("No valid answer words for accuracy calculation")
            return 0.0
        
        accuracy_ratio = overlap / total_answer_words
        result = min(1.0, accuracy_ratio * 1.2)
        logger.debug(f"Answer accuracy: {result}, overlap: {overlap}, total words: {total_answer_words}")
        return result
    
    def _clean_query(self, query: str) -> str:
        logger.debug(f"Cleaning query: {query[:50]}...")
        cleaned = re.sub(r'\s+', ' ', query.strip())
        cleaned = re.sub(r'[^\u0600-\u06FF\s\?\.\!،]', '', cleaned)
        logger.debug(f"Cleaned query: {cleaned[:50]}...")
        return cleaned
    
    def _calculate_enhanced_confidence(self, context_chunks: List[str], query: str, 
                                     query_context: QueryContext) -> float:
        logger.debug(f"Calculating confidence for query: {query[:50]}...")
        if not context_chunks:
            logger.debug("No context chunks for confidence calculation")
            return 0.0
        
        chunk_embeddings = self.generate_embeddings_batch(context_chunks)
        query_embedding = self.generate_embeddings_batch([query])
        
        similarities = []
        for chunk_emb in chunk_embeddings:
            sim = np.dot(query_embedding[0], chunk_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_emb)
            )
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        context_boost = 0.1 if len(query_context.related_history) > 0 else 0.0
        type_boost = 0.1 if query_context.query_type != "general" else 0.0
        
        confidence = float(avg_similarity) + context_boost + type_boost
        confidence = max(0.0, min(1.0, confidence))
        logger.debug(f"Calculated confidence: {confidence}, avg_similarity: {avg_similarity}")
        return confidence
    
    def _generate_fallback_response(self, context_text: str, query: str, query_context: QueryContext) -> str:
        logger.info(f"Generating fallback response for query: {query[:50]}...")
        if not context_text.strip():
            logger.debug("No context text for fallback response")
            return "معذرت، سوال جو جواب موجود ڊيٽا ۾ نه مليو آهي۔"
        
        query_lower = query.lower()
        context_lower = context_text.lower()
        
        sentences = re.split(r'[۔؟!]+', context_text)
        relevant_sentences = []
        
        query_words = set(query.split())
        for sentence in sentences:
            if sentence.strip():
                sentence_words = set(sentence.split())
                if len(query_words.intersection(sentence_words)) > 0:
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            response = "۔ ".join(relevant_sentences[:3])
            if len(response) > 500:
                response = response[:500] + "..."
            logger.debug(f"Generated fallback response with {len(relevant_sentences)} sentences")
            return response + "۔"
        
        first_part = context_text[:300]
        if len(context_text) > 300:
            first_part += "..."
        logger.debug("Using first 300 chars of context for fallback")
        return first_part
    
    def get_enhanced_response(self, query: str) -> Dict[str, any]:
        logger.info(f"Generating enhanced response for query: {query[:50]}...")
        try:
            cache_key = f"{query}_{len(self.conversation_memory.conversation_history)}"
            cache_key_hash = self.response_cache._get_cache_key(cache_key)
            cached_response = self.response_cache.get(cache_key_hash)
            
            if cached_response:
                logger.debug(f"Returning cached response for key: {cache_key_hash}")
                return cached_response
            
            query_context = self.enhanced_query_processing(query)
            retrieval_results = self.precision_retrieval(query_context)
            
            if not retrieval_results:
                logger.warning("No retrieval results found")
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
            
            selected_indices = []
            for chunk_info, _ in retrieval_results[:FINAL_CONTEXT_K]:
                for i, chunk in enumerate(self.chunk_data):
                    if chunk['text'] == chunk_info['text']:
                        selected_indices.append(i)
                        break
            
            expanded_indices = self.expand_context_with_neighbors(selected_indices)
            context_chunks = [self.chunk_data[i]['text'] for i in expanded_indices]
            context_text = "\n\n".join(context_chunks)
            logger.debug(f"Assembled context with {len(context_chunks)} chunks")
            
            confidence = self._calculate_enhanced_confidence(context_chunks, query, query_context)
            logger.debug(f"Calculated confidence: {confidence}")
            
            formatted_prompt = self.prompt.format(
                question=query,
                context=context_text,
                confidence=f"{confidence:.2f}",
                conversation_context=query_context.conversation_context
            )
            
            logger.info("Generating AI response")
            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[
                        {"role": "system", "content": "توھان شاھ عبداللطيف ڀٽائي جي معلومات جي باري ۾ ماھر آھيو۔ صرف ڄاڻايل حوالن مان معلومات ڏيو۔"},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600,
                    timeout=10
                )
                
                answer = response.choices[0].message.content
                logger.info("AI response generated successfully")
                
            except Exception as e:
                logger.error(f"Error calling Together API: {e}")
                answer = self._generate_fallback_response(context_text, query, query_context)
                logger.info("Used fallback response due to API error")
            
            context_chunks_for_verification = [self.chunk_data[i]['text'] for i in expanded_indices]
            accuracy_score = self.verify_answer_accuracy(answer, context_chunks_for_verification)
            final_confidence = min(confidence, accuracy_score)
            logger.debug(f"Final confidence: {final_confidence}, accuracy: {accuracy_score}")
            
            del context_chunks_for_verification, context_text, formatted_prompt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
            
            context_chunks_for_memory = [self.chunk_data[i]['text'] for i in expanded_indices]
            self.conversation_memory.add_turn(
                query=query,
                answer=answer,
                context_chunks=context_chunks_for_memory,
                confidence=final_confidence,
                chunk_indices=expanded_indices
            )
            logger.debug("Added response to conversation memory")
            
            self.response_cache.set(cache_key_hash, result)
            logger.debug(f"Cached response with key: {cache_key_hash}")
            
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
    logger.info("Getting RAG system instance")
    global _rag_system
    if _rag_system is None:
        _rag_system = ProductionRAGSystem()
    return _rag_system

def query_general_chatbot(query: str) -> str:
    logger.info(f"Querying general chatbot: {query[:50]}...")
    try:
        rag_system = get_rag_system()
        result = rag_system.get_enhanced_response(query)
        logger.debug(f"Chatbot response generated, confidence: {result['confidence']}")
        return result['answer']
        
    except Exception as e:
        logger.error(f"Error in general chatbot: {str(e)}")
        return f"معذرت، خرابي آئي آهي: {str(e)}"

def query_general_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    logger.info(f"Querying session-aware chatbot: {query[:50]}..., user_id: {user_id}, session_id: {session_id}")
    from ...services.session_service import SessionService
    
    try:
        if session_id:
            session = SessionService.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return {
                    'error': 'Session not found',
                    'code': 'SESSION_NOT_FOUND'
                }
            
            if not SessionService.verify_session_belongs_to_user(session_id, user_id):
                logger.warning(f"Session unauthorized for user {user_id}: {session_id}")
                return {
                    'error': 'Session does not belong to user',
                    'code': 'SESSION_UNAUTHORIZED'
                }
        else:
            session_name = query[:100] if len(query) <= 100 else query[:97] + "..."
            session_id = SessionService.create_session(user_id, session_name)
            logger.debug(f"Created new session: {session_id}")
        
        user_message_id = SessionService.save_message(session_id, 'user', query)
        logger.debug(f"Saved user message: {user_message_id}")
        
        SessionService.update_session_activity(session_id)
        logger.debug(f"Updated session activity: {session_id}")
        
        rag_system = get_rag_system()
        result = rag_system.get_enhanced_response(query)
        logger.debug(f"Generated chatbot response, confidence: {result['confidence']}")
        
        bot_message_id = SessionService.save_message(session_id, 'bot', result['answer'])
        logger.debug(f"Saved bot message: {bot_message_id}")
        
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
        logger.info(f"Returning session-aware response, session_id: {session_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error in session-aware general chatbot: {str(e)}")
        return {
            'error': f"معذرت، خرابي آئي آهي: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }