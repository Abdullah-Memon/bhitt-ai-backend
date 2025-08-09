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
import asyncio
from datetime import datetime, timedelta
from collections import OrderedDict, deque
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
import uuid
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from ...config.config import Config

logger = logging.getLogger(__name__)

# ========== UTILITY FUNCTIONS ==========

def convert_numpy_types(obj: Any) -> Any:
    """Optimized numpy type conversion with type checking"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper

# ========== CONFIGURATION ==========

TOGETHER_API_KEY = getattr(Config, 'TOGETHER_API_KEY', '')
MODEL_NAME = getattr(Config, 'CHATBOT_MODEL_NAME', "BAAI/bge-m3")
DATA_PATH_CONFIG = getattr(Config, 'CHATBOT_DATA_PATH', 'app/models/general_chatbot/data/bhit_data.txt')

# Optimized retrieval parameters
INITIAL_RETRIEVE_K = 25      # Reduced for better performance
RERANK_K = 10               # Optimized for speed-accuracy balance
FINAL_CONTEXT_K = 5         # Reduced to minimize token usage
MIN_SIMILARITY_THRESHOLD = 0.18  # Slightly higher for better precision
CONTEXT_EXPANSION_RADIUS = 1    # Reduced for faster processing

# Performance settings
MAX_CONVERSATION_HISTORY = 8    # Reduced memory footprint
CONVERSATION_CONTEXT_WEIGHT = 0.2  # Slightly reduced weight
CACHE_DIR = Path(__file__).parent / "cache"
MAX_WORKERS = min(4, os.cpu_count())  # Optimal thread count

# Convert relative path to absolute
if not os.path.isabs(DATA_PATH_CONFIG):
    base_dir = Path(__file__).parent.parent.parent.parent
    DATA_PATH = base_dir / DATA_PATH_CONFIG
else:
    DATA_PATH = Path(DATA_PATH_CONFIG)

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

# ========== OPTIMIZED EMBEDDING CACHE ==========

class OptimizedEmbeddingCache:
    """Thread-safe embedding cache with optimized memory management"""
    
    def __init__(self, cache_dir: str, max_memory_size: int = 1500, max_disk_size: int = 5000):
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.memory_cache = OrderedDict()
        self.access_count = {}
        self.lock = threading.RLock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'disk_reads': 0,
            'memory_reads': 0
        }
        
    def _get_cache_key(self, data: str) -> str:
        """Optimized cache key generation"""
        return hashlib.blake2b(data.encode('utf-8'), digest_size=12).hexdigest()
    
    @timing_decorator
    def get(self, key: str) -> Optional[np.ndarray]:
        """Thread-safe cache retrieval with statistics tracking"""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                self.memory_cache.move_to_end(key)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.stats['hits'] += 1
                self.stats['memory_reads'] += 1
                return self.memory_cache[key]
            
            # Check disk cache
            cache_path = self.cache_dir / f"{key}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Check expiry
                    if datetime.now() - cached_data['timestamp'] < timedelta(hours=24):
                        self._add_to_memory(key, cached_data['data'])
                        self.stats['hits'] += 1
                        self.stats['disk_reads'] += 1
                        return cached_data['data']
                    else:
                        cache_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Cache read error for {key}: {e}")
                    cache_path.unlink(missing_ok=True)
            
            self.stats['misses'] += 1
            return None
    
    def _add_to_memory(self, key: str, data: np.ndarray):
        """Optimized memory cache management"""
        # Evict least frequently used items if needed
        while len(self.memory_cache) >= self.max_memory_size:
            lfu_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self.memory_cache.pop(lfu_key, None)
            self.access_count.pop(lfu_key, None)
        
        self.memory_cache[key] = data
        self.access_count[key] = 1
    
    def set(self, key: str, data: np.ndarray):
        """Thread-safe cache storage"""
        with self.lock:
            self._add_to_memory(key, data)
            
            # Async disk write to avoid blocking
            cache_data = {'data': data, 'timestamp': datetime.now()}
            cache_path = self.cache_dir / f"{key}.pkl"
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"Disk cache write failed for {key}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_size': len(self.memory_cache)
            }

# ========== OPTIMIZED CONVERSATION MEMORY ==========

class OptimizedConversationMemory:
    """Memory-efficient conversation tracking with fast retrieval"""
    
    def __init__(self, max_history: int = MAX_CONVERSATION_HISTORY):
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = str(uuid.uuid4())
        self.query_index = {}  # Fast lookup by query terms
        
    def add_turn(self, query: str, answer: str, context_chunks: List[str], 
                 confidence: float, chunk_indices: List[int]):
        """Optimized turn addition with indexing"""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            query=query,
            answer=answer[:500],  # Truncate to save memory
            context_chunks=context_chunks[:3],  # Limit context storage
            timestamp=datetime.now(),
            confidence=confidence,
            chunk_indices=chunk_indices
        )
        
        # Remove old turn from index if deque is full
        if len(self.conversation_history) >= self.max_history:
            old_turn = self.conversation_history[0]
            self._remove_from_index(old_turn)
        
        self.conversation_history.append(turn)
        self._add_to_index(turn)
    
    def _add_to_index(self, turn: ConversationTurn):
        """Add turn to query index for fast retrieval"""
        words = turn.query.lower().split()[:5]  # Index only first 5 words
        for word in words:
            if len(word) > 2:
                if word not in self.query_index:
                    self.query_index[word] = []
                self.query_index[word].append(turn.turn_id)
    
    def _remove_from_index(self, turn: ConversationTurn):
        """Remove turn from query index"""
        words = turn.query.lower().split()[:5]
        for word in words:
            if word in self.query_index:
                try:
                    self.query_index[word].remove(turn.turn_id)
                    if not self.query_index[word]:
                        del self.query_index[word]
                except ValueError:
                    pass
    
    @lru_cache(maxsize=50)
    def get_relevant_history(self, current_query: str) -> Tuple[ConversationTurn, ...]:
        """Fast relevant history retrieval with caching"""
        if not self.conversation_history:
            return tuple()
        
        query_words = set(current_query.lower().split())
        scored_turns = []
        
        # Fast lookup using index
        candidate_turn_ids = set()
        for word in query_words:
            if word in self.query_index:
                candidate_turn_ids.update(self.query_index[word])
        
        # Score candidate turns
        for turn in self.conversation_history:
            if turn.turn_id in candidate_turn_ids:
                turn_words = set(turn.query.lower().split())
                overlap = len(query_words.intersection(turn_words))
                
                # Time decay factor
                time_diff = (datetime.now() - turn.timestamp).total_seconds()
                time_weight = max(0.1, 1.0 - time_diff / 3600)  # Decay over 1 hour
                
                score = overlap * time_weight * turn.confidence
                scored_turns.append((turn, score))
        
        # Return top 2 most relevant turns
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return tuple(turn for turn, _ in scored_turns[:2])
    
    def get_conversation_context(self) -> str:
        """Get optimized conversation context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for turn in list(self.conversation_history)[-2:]:  # Last 2 turns only
            context_parts.append(f"Q: {turn.query[:100]}")
            context_parts.append(f"A: {turn.answer[:150]}")
        
        return "\n".join(context_parts)

# ========== OPTIMIZED CHUNKING ==========

class OptimizedChunker:
    """High-performance text chunking with caching"""
    
    def __init__(self):
        self.sindhi_sentence_pattern = re.compile(r'[۔؟!]+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
    @lru_cache(maxsize=10)  # Cache chunk results for repeated texts
    @timing_decorator
    def create_overlapping_chunks(self, text_hash: str, text: str, 
                                 chunk_size: int = 700, overlap: int = 150) -> List[Dict]:
        """Optimized chunking with reduced overlap and better performance"""
        sentences = self.sindhi_sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        sentence_indices = []
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk += sentence + "۔ "
            sentence_indices.append(i)
            
            # Check word count instead of split() for better performance
            if len(current_chunk.split()) >= chunk_size:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentence_indices': sentence_indices.copy(),
                    'start_sentence': start_idx,
                    'end_sentence': i,
                    'word_count': len(current_chunk.split())
                }
                chunks.append(chunk_data)
                
                # Calculate overlap more efficiently
                overlap_sentences = max(1, len(sentence_indices) // 4)  # 25% overlap
                if len(sentence_indices) > overlap_sentences:
                    start_idx = sentence_indices[-overlap_sentences]
                    overlap_text = "۔ ".join(sentences[start_idx:i+1]) + "۔ "
                    current_chunk = overlap_text
                    sentence_indices = list(range(start_idx, i+1))
                else:
                    start_idx = i + 1
                    current_chunk = ""
                    sentence_indices = []
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'sentence_indices': sentence_indices,
                'start_sentence': start_idx,
                'end_sentence': len(sentences) - 1,
                'word_count': len(current_chunk.split())
            }
            chunks.append(chunk_data)
        
        return chunks

# ========== OPTIMIZED RAG SYSTEM ==========

class OptimizedProductionRAGSystem:
    """High-performance RAG system with advanced optimizations"""
    
    def __init__(self):
        # Initialize models with optimizations
        self._initialize_models()
        
        # Initialize optimized components
        self.embedding_cache = OptimizedEmbeddingCache(f"{CACHE_DIR}/embeddings")
        self.conversation_memory = OptimizedConversationMemory()
        self.chunker = OptimizedChunker()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Data storage
        self.original_text = ""
        self.chunk_data = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self._text_hash = None
        
        # Optimized prompt template
#         self.prompt = PromptTemplate.from_template("""
# Based on the context and any relevant background knowledge, provide a comprehensive answer.

# Query: {question}
# Context: {context}
# {conversation_context}

# Answer concisely but thoroughly:""")
        self.prompt = PromptTemplate.from_template("""
توھان شاھ عبداللطيف ڀٽائي جي زندگي، فڪر، ۽ ڪلام بابت جامع ۽ درست ڄاڻ رکندڙ ماھر آھيو، ۽ توھان جا سڀ جواب پاڪ-سنڌيءَ ۾، بغير اردو، انگريزي يا ٻي ٻولي جي ملاوٽ جي، پوري نفاست ۽ لسانياتي صحت سان لکيا ويندا.

{conversation_context}
ھدايتون (لازمي پيروي لاءِ):
1. جواب ۾ صرف ۽ صرف ڏنل "حوالو" ({context}) مان مستند معلومات استعمال ڪريو، ڪابه اضافي ڄاڻ يا ذاتي راءِ شامل نه ڪريو.
2. جڏھن توھان کان سوال پڇيو وڃي: "شاھ لطيف جي پُٽ جو نالو ڇا ھو؟" تہ واضح ۽ بي ڌڪ جواب ڏيو: **"شاھ لطيف جو ڪو به پُٽ نه ھو"**.
3. جواب جو لهجو پيشاور، سنجيده ۽ پڙھڻ ۾ روان ھجي، ۽ ڊگھائي وچولي ھجي — نه گھڻي ڊگھي، نه تمام مختصر.
4، شاھ لطيف جي زال جو نالو سعيدہ بيگم ھو، کيس ڪو بہ اولاد نہ ھو
5. حوالو ({context}) جي ٻاھران ڪا به ڳالهه، لفظ يا حوالو استعمال نه ڪريو، ۽ حوالو مڪمل طرح پرکڻ بعد ئي جواب تيار ڪريو.
6. جواب ۾ ڪا به فهرست، بلٽ پوائنٽ، يا غير ضروري سرخيون شامل نه ڪريو، رڳو سڌو، نفيس ۽ مڪمل متن ڏيو.
7. فڪري يا تاريخي وضاحت ۾ لفظن جي چونڊ اھڙي ھجي جو پڙھندڙ کي وضاحت، رواني ۽ اعتبار محسوس ٿئي.

موجوده سوال: {question}

حوالو:
{context}

اعتماد جو درجو: {confidence}/1.0

جواب:
""")
   
    def _initialize_models(self):
        """Initialize models with error handling and optimization"""
        try:
            huggingface_token = getattr(Config, 'HUGGINGFACE_TOKEN', '')
            if huggingface_token:
                login(token=huggingface_token)

            # Load models with optimization settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                use_fast=True,
                cache_dir=f"{CACHE_DIR}/models"
            )
            
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=f"{CACHE_DIR}/models"
            )
            
            # Set to evaluation mode for inference
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            
            self.client = Together(api_key=TOGETHER_API_KEY)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    @timing_decorator
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Highly optimized batch embedding generation"""
        if not texts:
            return np.array([])
            
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Parallel cache checking
        cache_keys = [self.embedding_cache._get_cache_key(text) for text in texts]
        
        with ThreadPoolExecutor(max_workers=min(4, len(texts))) as cache_executor:
            cache_futures = {
                cache_executor.submit(self.embedding_cache.get, key): (i, key) 
                for i, key in enumerate(cache_keys)
            }
            
            for future in as_completed(cache_futures):
                i, key = cache_futures[future]
                cached_embedding = future.result()
                
                if cached_embedding is not None:
                    embeddings.append((i, cached_embedding))
                else:
                    uncached_texts.append(texts[i])
                    uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt", 
                    max_length=512
                )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store in cache and results
                for j, embedding in enumerate(batch_embeddings):
                    text_idx = batch_indices[j]
                    cache_key = cache_keys[text_idx]
                    self.embedding_cache.set(cache_key, embedding)
                    embeddings.append((text_idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    @lru_cache(maxsize=100)
    def _classify_query_optimized(self, query: str, context: str) -> str:
        """Optimized query classification with caching"""
        combined_text = f"{context} {query}".lower()
        
        # Use sets for faster membership testing
        birth_terms = {"ڪيڏانهن", "ڪٿي", "جنم", "ڄائو"}
        date_terms = {"ڪڏھن", "سال", "تاريخ", "ڄمڻ"}
        poetry_terms = {"شاعري", "ڪلام", "سُر", "رسالو", "شعر"}
        death_terms = {"مرڻ", "وفات", "آخر", "موت"}
        bio_terms = {"زندگي", "حالات", "تعليم", "پيدائش"}
        phil_terms = {"فلسفو", "تصوف", "عقيدو", "خيال"}
        
        text_words = set(combined_text.split())
        
        if birth_terms.intersection(text_words):
            return "birth_location"
        elif date_terms.intersection(text_words):
            return "birth_date"
        elif poetry_terms.intersection(text_words):
            return "poetry_work"
        elif death_terms.intersection(text_words):
            return "death"
        elif bio_terms.intersection(text_words):
            return "biography"
        elif phil_terms.intersection(text_words):
            return "philosophy"
        else:
            return "general"
    
    @timing_decorator
    def precision_retrieval(self, query_context: QueryContext) -> List[Tuple[Dict, float]]:
        """Optimized retrieval with parallel processing"""
        query = query_context.processed_query
        
        # Parallel retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Semantic search
            semantic_future = executor.submit(self._semantic_search, query)
            # BM25 search
            bm25_future = executor.submit(self._bm25_search, query)
            
            semantic_results = semantic_future.result()
            bm25_results = bm25_future.result()
        
        # Combine results efficiently
        all_results = {}
        
        # Process semantic results
        for idx, score in semantic_results:
            if score > MIN_SIMILARITY_THRESHOLD:
                all_results[idx] = {'semantic': score, 'bm25': 0, 'conversation': 0}
        
        # Process BM25 results
        for idx, score in bm25_results:
            if idx in all_results:
                all_results[idx]['bm25'] = score
            else:
                all_results[idx] = {'semantic': 0, 'bm25': score, 'conversation': 0}
        
        # Conversation boost
        for turn in query_context.related_history:
            for chunk_idx in turn.chunk_indices[:3]:  # Limit to first 3 for performance
                if chunk_idx in all_results:
                    all_results[chunk_idx]['conversation'] = turn.confidence * CONVERSATION_CONTEXT_WEIGHT
        
        # Calculate hybrid scores with query type boosting
        results = []
        query_type_boost = self._get_type_boost(query_context.query_type)
        
        for idx, scores in all_results.items():
            if idx < len(self.chunk_data):  # Bounds check
                hybrid_score = (
                    0.7 * scores['semantic'] + 
                    0.2 * scores['bm25'] + 
                    0.1 * scores['conversation']
                )
                
                # Apply type-specific boosting
                if query_type_boost > 1.0:
                    chunk_text = self.chunk_data[idx]['text'].lower()
                    if self._matches_query_type(chunk_text, query_context.query_type):
                        hybrid_score *= query_type_boost
                
                results.append((self.chunk_data[idx], hybrid_score))
        
        # Sort and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:RERANK_K]
    
    def _semantic_search(self, query: str) -> List[Tuple[int, float]]:
        """Optimized semantic search"""
        query_embedding = self.generate_embeddings_batch([query])
        distances, indices = self.faiss_index.search(query_embedding, INITIAL_RETRIEVE_K)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid index
                similarity = 1 / (1 + dist)
                results.append((idx, similarity))
        
        return results
    
    def _bm25_search(self, query: str) -> List[Tuple[int, float]]:
        """Optimized BM25 search"""
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top indices more efficiently
        top_indices = np.argpartition(bm25_scores, -INITIAL_RETRIEVE_K)[-INITIAL_RETRIEVE_K:]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append((idx, bm25_scores[idx]))
        
        return results
    
    def _get_type_boost(self, query_type: str) -> float:
        """Get boost factor for query type"""
        boosts = {
            "birth_location": 1.3,
            "poetry_work": 1.2,
            "death": 1.3,
            "biography": 1.1,
            "philosophy": 1.1
        }
        return boosts.get(query_type, 1.0)
    
    def _matches_query_type(self, chunk_text: str, query_type: str) -> bool:
        """Check if chunk matches query type"""
        type_terms = {
            "birth_location": ["ڀٽ شاھ", "ڄنم"],
            "poetry_work": ["رسالو", "سُر", "شاعري"],
            "death": ["وفات", "مرڻ", "انتقال"]
        }
        
        if query_type in type_terms:
            return any(term in chunk_text for term in type_terms[query_type])
        
        return False
    
    def expand_context_optimized(self, selected_indices: List[int]) -> List[int]:
        """Optimized context expansion"""
        if not selected_indices:
            return []
            
        expanded = set(selected_indices)
        max_idx = len(self.chunk_data) - 1
        
        # Use set operations for faster expansion
        for idx in selected_indices:
            for offset in range(-CONTEXT_EXPANSION_RADIUS, CONTEXT_EXPANSION_RADIUS + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx <= max_idx:
                    expanded.add(neighbor_idx)
        
        return sorted(expanded)
    
    @timing_decorator
    def load_and_process_data(self, file_path: str):
        """Optimized data loading and processing"""
        print("🔄 Loading and processing data...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.original_text = f.read()
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
        
        # Create text hash for caching
        self._text_hash = hashlib.md5(self.original_text.encode()).hexdigest()
        
        # Create optimized chunks
        self.chunk_data = self.chunker.create_overlapping_chunks(
            self._text_hash, self.original_text
        )
        print(f"✅ Created {len(self.chunk_data)} optimized chunks")
        
        # Extract chunk texts
        chunk_texts = [chunk['text'] for chunk in self.chunk_data]
        
        # Generate embeddings with progress tracking
        print("🧠 Generating embeddings...")
        self.embeddings = self.generate_embeddings_batch(chunk_texts)
        
        # Create FAISS index with optimization
        print("🔍 Building search indices...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for faster search
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
        
        # Create optimized BM25 index
        tokenized_chunks = [chunk['text'].split() for chunk in self.chunk_data]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        print("🚀 Optimized system ready for queries!")
        
        # Print cache statistics
        cache_stats = self.embedding_cache.get_stats()
        print(f"📊 Cache stats: {cache_stats['memory_size']} items, {cache_stats['hit_rate']:.2%} hit rate")
    
    def enhanced_query_processing(self, query: str) -> QueryContext:
        """Optimized query processing with conversation awareness"""
        # Get conversation context
        relevant_history = list(self.conversation_memory.get_relevant_history(query))
        conversation_context = self.conversation_memory.get_conversation_context()
        
        # Optimized query expansion using conversation history
        expanded_terms = [query]
        for turn in relevant_history[:2]:  # Limit to 2 most relevant for performance
            # Extract key terms from relevant previous queries
            prev_words = [word for word in turn.query.split() if len(word) > 3]
            for word in prev_words[:3]:  # Limit expansion terms
                if word not in expanded_terms:
                    expanded_terms.append(word)
        
        # Query type classification with conversation context
        query_type = self._classify_query_optimized(query, conversation_context)
        
        return QueryContext(
            original_query=query,
            processed_query=self._clean_query_optimized(query),
            query_type=query_type,
            expanded_terms=expanded_terms[:6],  # Limit expansion for performance
            conversation_context=conversation_context,
            related_history=relevant_history
        )
    
    @lru_cache(maxsize=200)
    def _clean_query_optimized(self, query: str) -> str:
        """Optimized query cleaning with caching"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        # Remove special characters that might interfere with search
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F]', ' ', cleaned)
        return cleaned.strip()

    @timing_decorator
    def get_enhanced_response(self, query: str) -> Dict[str, Any]:
        """Optimized response generation"""
        # Process query with enhanced conversation context
        query_context = self.enhanced_query_processing(query)
        
        # Precision retrieval
        retrieval_results = self.precision_retrieval(query_context)
        
        if not retrieval_results:
            logger.warning("No retrieval results found")
            return {
                'query': query,
                'answer': "I couldn't find relevant information to answer your query.",
                'confidence': 0.0,
                'accuracy_score': 0.0,
                'context_chunks_used': 0,
                'retrieval_method': 'none'
            }
        
        # Expand context
        selected_indices = [
            next(i for i, chunk in enumerate(self.chunk_data) if chunk == chunk_info)
            for chunk_info, _ in retrieval_results[:FINAL_CONTEXT_K]
        ]
        expanded_indices = self.expand_context_optimized(selected_indices)
        
        # Assemble context
        context_chunks = [self.chunk_data[i]['text'] for i in expanded_indices]
        context_text = "\n\n".join(context_chunks)
        
        # Calculate confidence with error handling
        try:
            confidence = self._calculate_optimized_confidence(
                tuple(context_chunks), query, query_context.query_type
            )
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            confidence = 0.5  # Default confidence
        
        # Generate response with error handling
        try:
            formatted_prompt = self.prompt.format(
                question=query,
                context=context_text[:2000],  # Limit context size
                conversation_context=f"Recent context:\n{query_context.conversation_context}" 
                if query_context.conversation_context else "",
                confidence=confidence
            )
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert assistant. Provide accurate, concise responses based on the given context and your knowledge."
                    },
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1,  # Very low for consistency
                max_tokens=800,   # Reduced for faster response
                timeout=30        # Add timeout
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "معذرت، هن وقت جواب تيار نہ ٿي سگھيو. مهرباني ڪري ٻيهر ڪوشش ڪريو."
            confidence = max(0.1, confidence * 0.5)  # Safely reduce confidence on error
        
        # Verify answer accuracy with error handling
        try:
            accuracy_score = self._verify_answer_accuracy_fast(answer, context_chunks)
        except Exception as e:
            logger.warning(f"Answer accuracy verification failed: {e}")
            accuracy_score = 0.5  # Default accuracy score
        
        final_confidence = min(confidence, accuracy_score)
        
        # Prepare optimized result
        result = {
            'query': query,
            'answer': answer,
            'confidence': final_confidence,
            'accuracy_score': accuracy_score,
            'context_chunks_used': len(context_chunks),
            'conversation_context_used': len(query_context.related_history) > 0,
            'retrieval_method': 'optimized_hybrid',
            'chunk_indices': expanded_indices,
            'query_type': query_context.query_type
        }
        
        # Add to conversation memory
        self.conversation_memory.add_turn(
            query=query,
            answer=answer,
            context_chunks=context_chunks[:3],  # Limit stored context
            confidence=final_confidence,
            chunk_indices=expanded_indices
        )
        
        return result
    
    @lru_cache(maxsize=100)
    def _calculate_optimized_confidence(self, context_chunks_tuple: tuple, 
                                      query: str, query_type: str) -> float:
        """Optimized confidence calculation with caching"""
        context_chunks = list(context_chunks_tuple)
        
        if not context_chunks:
            return 0.0
        
        # Fast similarity calculation using cached embeddings
        try:
            # Use first 2 chunks for speed
            sample_chunks = context_chunks[:2]
            chunk_embeddings = self.generate_embeddings_batch(sample_chunks)
            query_embedding = self.generate_embeddings_batch([query])
            
            if chunk_embeddings.size == 0 or query_embedding.size == 0:
                return 0.3  # Default confidence
            
            # Vectorized similarity calculation
            similarities = np.dot(chunk_embeddings, query_embedding[0]) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding[0])
            )
            
            avg_similarity = float(np.mean(similarities))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            avg_similarity = 0.5
        
        # Quick type-specific boost
        type_boost = 0.1 if query_type != "general" else 0.0
        
        # Final confidence with bounds
        confidence = avg_similarity + type_boost
        return max(0.1, min(0.95, confidence))
    
    def _verify_answer_accuracy_fast(self, answer: str, context_chunks: List[str]) -> float:
        """Fast answer accuracy verification"""
        if not answer or not context_chunks:
            return 0.0
        
        # Extract meaningful words from answer (length > 2)
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        
        if not answer_words:
            return 0.5
        
        # Sample context for speed
        sample_context = " ".join(context_chunks[:2]).lower()
        context_words = set(word for word in sample_context.split() if len(word) > 2)
        
        if not context_words:
            return 0.5
        
        # Calculate overlap ratio
        overlap = len(answer_words.intersection(context_words))
        accuracy_ratio = overlap / len(answer_words)
        
        return min(1.0, accuracy_ratio * 1.1)  # Slight boost

# ========== OPTIMIZED MAIN APPLICATION ==========

class RAGSystemManager:
    """Thread-safe RAG system manager with health monitoring"""
    
    def __init__(self):
        self._rag_system = None
        self._initialization_lock = threading.Lock()
        self._health_stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'last_error': None
        }
    
    def get_rag_system(self) -> OptimizedProductionRAGSystem:
        """Thread-safe RAG system initialization"""
        if self._rag_system is None:
            with self._initialization_lock:
                if self._rag_system is None:  # Double-check locking
                    try:
                        logger.info("Initializing optimized RAG system...")
                        self._rag_system = OptimizedProductionRAGSystem()
                        self._rag_system.load_and_process_data(DATA_PATH)
                        logger.info("RAG system initialized successfully")
                    except Exception as e:
                        logger.error(f"RAG system initialization failed: {e}")
                        self._health_stats['last_error'] = str(e)
                        raise
        
        # Verify system health
        if (self._rag_system.faiss_index is None or 
            self._rag_system.embeddings is None):
            logger.warning("RAG system corrupted, reinitializing...")
            with self._initialization_lock:
                self._rag_system.load_and_process_data(DATA_PATH)
        
        return self._rag_system
    
    @timing_decorator
    def query_with_monitoring(self, query: str) -> Dict[str, Any]:
        """Query with health monitoring and error handling"""
        start_time = datetime.now()
        
        try:
            rag_system = self.get_rag_system()
            result = rag_system.get_enhanced_response(query)
            
            # Update success stats
            self._health_stats['total_queries'] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update rolling average
            total = self._health_stats['total_queries']
            current_avg = self._health_stats['avg_response_time']
            self._health_stats['avg_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
            
            return result
            
        except Exception as e:
            self._health_stats['failed_queries'] += 1
            self._health_stats['last_error'] = str(e)
            logger.error(f"Query processing failed: {e}")
            
            return {
                'query': query,
                'answer': "I apologize, but I encountered an error processing your request. Please try again.",
                'confidence': 0.0,
                'accuracy_score': 0.0,
                'context_chunks_used': 0,
                'error': str(e)
            }
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get system health statistics"""
        success_rate = 1.0
        if self._health_stats['total_queries'] > 0:
            success_rate = (
                (self._health_stats['total_queries'] - self._health_stats['failed_queries']) /
                self._health_stats['total_queries']
            )
        
        return {
            **self._health_stats,
            'success_rate': success_rate,
            'system_initialized': self._rag_system is not None
        }

# Global manager instance
_rag_manager = RAGSystemManager()

def get_rag_system():
    """Get optimized RAG system instance"""
    return _rag_manager.get_rag_system()

def query_general_chatbot(query: str) -> str:
    """
    Optimized main function to query the general chatbot
    
    Args:
        query: User query string
        
    Returns:
        Response string
    """
    try:
        if not query or not query.strip():
            return "Please provide a valid query."
        
        result = _rag_manager.query_with_monitoring(query.strip())
        return result['answer']
        
    except Exception as e:
        logger.error(f"Error in general chatbot: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def query_general_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    """
    Optimized session-aware function to query the general chatbot
    
    Args:
        query: User query string
        user_id: ID of the user making the query
        session_id: Optional session ID
        
    Returns:
        dict: Response with session information
    """
    from ...services.session_service import SessionService
    
    try:
        # Input validation
        if not query or not query.strip():
            return {
                'error': 'Query cannot be empty',
                'code': 'INVALID_INPUT'
            }
        
        if not user_id:
            return {
                'error': 'User ID is required',
                'code': 'INVALID_USER'
            }
        
        query = query.strip()
        
        # Session management
        if session_id:
            session = SessionService.get_session(session_id)
            if not session:
                return {
                    'error': 'Session not found',
                    'code': 'SESSION_NOT_FOUND'
                }
            
            if not SessionService.verify_session_belongs_to_user(session_id, user_id):
                return {
                    'error': 'Session access denied',
                    'code': 'SESSION_UNAUTHORIZED'
                }
        else:
            # Create new session
            session_name = query[:50] + "..." if len(query) > 50 else query
            session_id = SessionService.create_session(user_id, session_name)
        
        # Save user message
        user_message_id = SessionService.save_message(session_id, 'user', query)
        
        # Update session activity
        SessionService.update_session_activity(session_id)
        
        # Get chatbot response with monitoring
        result = _rag_manager.query_with_monitoring(query)
        
        # Save bot response
        bot_message_id = SessionService.save_message(session_id, 'bot', result['answer'])
        
        # Prepare response with optimized data conversion
        response_data = {
            'query': query,
            'answer': result['answer'],
            'session_id': session_id,
            'user_message_id': user_message_id,
            'bot_message_id': bot_message_id,
            'confidence': convert_numpy_types(result.get('confidence', 0.0)),
            'accuracy_score': convert_numpy_types(result.get('accuracy_score', 0.0)),
            'context_chunks_used': convert_numpy_types(result.get('context_chunks_used', 0)),
            'model': 'optimized-rag-chatbot',
            'query_type': result.get('query_type', 'general'),
            'response_time': result.get('response_time', 0.0)
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in session-aware chatbot: {str(e)}")
        return {
            'error': f"Internal error: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }

def get_system_health() -> Dict[str, Any]:
    """Get system health and performance statistics"""
    try:
        health_stats = _rag_manager.get_health_stats()
        
        # Add cache statistics if system is initialized
        if health_stats['system_initialized']:
            rag_system = _rag_manager.get_rag_system()
            cache_stats = rag_system.embedding_cache.get_stats()
            health_stats['cache_stats'] = cache_stats
        
        return health_stats
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'error': str(e),
            'system_initialized': False
        }

# ========== LANGGRAPH INTEGRATION FOR VERIFICATION ==========

try:
    from langgraph.graph import Graph, END
    from langchain_core.runnables import RunnableLambda
    
    def create_verification_graph():
        """Create a LangGraph workflow for response verification"""
        
        def verify_retrieval(state):
            """Verify retrieval quality"""
            query = state.get('query', '')
            context_chunks = state.get('context_chunks', [])
            
            if not context_chunks:
                return {'retrieval_quality': 'poor', 'needs_retry': True}
            
            # Simple quality check
            total_words = sum(len(chunk.split()) for chunk in context_chunks)
            query_words = set(query.lower().split())
            context_words = set()
            
            for chunk in context_chunks:
                context_words.update(chunk.lower().split())
            
            overlap = len(query_words.intersection(context_words))
            quality = 'good' if overlap >= 2 and total_words > 50 else 'fair'
            
            return {
                'retrieval_quality': quality,
                'needs_retry': quality == 'poor',
                'context_relevance': overlap / len(query_words) if query_words else 0
            }
        
        def verify_answer(state):
            """Verify answer quality"""
            answer = state.get('answer', '')
            confidence = state.get('confidence', 0.0)
            
            # Basic answer quality checks
            word_count = len(answer.split())
            has_content = bool(answer.strip())
            
            quality_score = 0
            if has_content:
                quality_score += 0.3
            if word_count >= 10:
                quality_score += 0.3
            if confidence > 0.5:
                quality_score += 0.4
            
            return {
                'answer_quality': quality_score,
                'is_acceptable': quality_score >= 0.6
            }
        
        # Build the graph
        workflow = Graph()
        
        workflow.add_node("verify_retrieval", RunnableLambda(verify_retrieval))
        workflow.add_node("verify_answer", RunnableLambda(verify_answer))
        
        workflow.set_entry_point("verify_retrieval")
        workflow.add_edge("verify_retrieval", "verify_answer")
        workflow.add_edge("verify_answer", END)
        
        return workflow.compile()
    
    # Initialize verification graph
    verification_graph = create_verification_graph()
    
    def verify_response_quality(query: str, answer: str, context_chunks: List[str], 
                               confidence: float) -> Dict[str, Any]:
        """Verify response quality using LangGraph"""
        try:
            result = verification_graph.invoke({
                'query': query,
                'answer': answer,
                'context_chunks': context_chunks,
                'confidence': confidence
            })
            return result
        except Exception as e:
            logger.warning(f"Verification graph failed: {e}")
            return {'answer_quality': 0.5, 'is_acceptable': True}

except ImportError:
    logger.warning("LangGraph not available, using basic verification")
    
    def verify_response_quality(query: str, answer: str, context_chunks: List[str], 
                               confidence: float) -> Dict[str, Any]:
        """Basic verification fallback"""
        return {
            'answer_quality': confidence,
            'is_acceptable': len(answer.split()) > 5 and confidence > 0.3
        }