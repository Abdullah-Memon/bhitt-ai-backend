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
import os
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

# Enhanced retrieval parameters
INITIAL_RETRIEVE_K = 30      # Broader initial search
RERANK_K = 15               # More candidates for reranking
FINAL_CONTEXT_K = 6         # Expanded final context
MIN_SIMILARITY_THRESHOLD = 0.1  # Lower threshold to avoid missing relevant content
CONTEXT_EXPANSION_RADIUS = 2    # Adjacent chunks to include

# Conversational memory
MAX_CONVERSATION_HISTORY = 10
CONVERSATION_CONTEXT_WEIGHT = 0.3

# Cache settings
CACHE_DIR = Path(__file__).parent / "cache"

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

# ========== EMBEDDING CACHE ONLY ==========

class EmbeddingCache:
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
        
    def create_overlapping_chunks(self, text: str, chunk_size: int = 500, 
                                 overlap: int = 100) -> List[Dict]:
        """Create chunks with metadata and overlap tracking"""
        sentences = re.split(self.sindhi_sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        sentence_indices = []
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk += sentence + "۔ "
            sentence_indices.append(i)
            
            if len(current_chunk.split()) >= chunk_size:
                # Create chunk with metadata
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentence_indices': sentence_indices.copy(),
                    'start_sentence': start_idx,
                    'end_sentence': i,
                    'char_start': text.find(sentences[start_idx]),
                    'char_end': text.find(sentence) + len(sentence)
                }
                chunks.append(chunk_data)
                
                # Calculate overlap
                overlap_sentences = max(1, overlap // 50)  # Approx sentences for overlap
                if len(sentence_indices) > overlap_sentences:
                    start_idx = sentence_indices[-overlap_sentences]
                    current_chunk = "۔ ".join(sentences[start_idx:i+1]) + "۔ "
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
                'char_start': text.find(sentences[start_idx]) if sentence_indices else 0,
                'char_end': len(text)
            }
            chunks.append(chunk_data)
        
        return chunks

# ========== ENHANCED RAG SYSTEM ==========

class ProductionRAGSystem:
    def __init__(self):
        # Initialize models
        huggingface_token = getattr(Config, 'HUGGINGFACE_TOKEN', '')
        if huggingface_token:
            login(token=huggingface_token)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.client = Together(api_key=TOGETHER_API_KEY)
        
        # Initialize components - ONLY embedding cache now
        self.embedding_cache = EmbeddingCache(f"{CACHE_DIR}/embeddings")
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
توھان شاھ عبداللطيف ڀٽائي جي زندگي ۽ ڪم بابت تفصيلي ڄاڻ رکندا آھيو. مهرباني ڪري ھيٺين ھدايتن تي عمل ڪندي سوال جو جواب ڏيو:
ھدايتون:

جواب صرف ۽ صرف ڄاڻايل حوالي (Context) مان حاصل ڪيل معلومات تي ٻڌل ھجي.
جواب خالص سنڌي ٻولي ۾ ڏيو، ۽ اردو، انگريزي يا ڪنھن ٻي ٻولي جا لفظ استعمال نه ڪريو.
جيڪڏھن سوال “شاھ لطيف جو پٽ ڪير ھو؟” يا ان سان ملندڙ جلندڙ ھجي، ته جواب ڏيو: “شاھ لطيف جو ڪو به پٽ نه ھو.”
جيڪڏھن ڄاڻايل حوالي (Context) ۾ سوال جو جواب موجود نه ھجي، ته جواب ڏيو: “ھن سوال جو جواب حوالي ۾ موجود نه آھي.”
جواب جامع، درست ۽ صاف سٿرو ھجي، ۽ ضرورت کان وڌيڪ تفصيل کان پاسو ڪريو.
جيڪڏھن حوالو خالي ھجي يا سوال سان لاڳاپيل معلومات فراهم نه ڪري، ته چئو: “ھن سوال جو جواب حوالي ۾ موجود نه آھي.”
If someone uses abusive words or language say "مھرباني ڪري ادب سان سوال پڇو."
موجوده سوال: {question}
حوالو (Context):


""")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Optimized batch embedding generation with caching"""
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
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            inputs = self.tokenizer(uncached_texts, padding=True, truncation=True,
                                  return_tensors="pt", max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            new_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Cache new embeddings
            for i, embedding in enumerate(new_embeddings):
                text_idx = uncached_indices[i]
                cache_key = self.embedding_cache._get_cache_key(uncached_texts[i])
                self.embedding_cache.set(cache_key, embedding)
                embeddings.append((text_idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
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
    
    def _clean_query(self, query: str) -> str:
        """Enhanced query cleaning"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        # Remove non-Sindhi characters except basic punctuation
        cleaned = re.sub(r'[^\u0600-\u06FF\s\?\.!،]', '', cleaned)
        return cleaned
    
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
    
    def load_and_process_data(self, file_path: str):
        """Enhanced data loading and processing"""
        print("🔄 Loading and processing data...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
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
    
    def get_enhanced_response(self, query: str) -> Dict[str, any]:
        """Generate enhanced response with conversation memory - NO RESPONSE CACHE"""
        # Process query with conversation context
        query_context = self.enhanced_query_processing(query)
        
        # Precision retrieval
        retrieval_results = self.precision_retrieval(query_context)
        
        # Expand context with neighboring chunks
        selected_indices = [list(self.chunk_data).index(chunk_info) 
                          for chunk_info, _ in retrieval_results[:FINAL_CONTEXT_K]]
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
        
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "توھان شاھ عبداللطيف ڀٽائي جي معلومات جي باري ۾ ماھر آھيو۔ صرف ڄاڻايل حوالن مان معلومات ڏيو۔"},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.2,  # Lower temperature for accuracy
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Verify answer accuracy
        accuracy_score = self.verify_answer_accuracy(answer, context_chunks)
        final_confidence = min(confidence, accuracy_score)
        
        # Prepare result
        result = {
            'query': query,
            'answer': answer,
            'confidence': final_confidence,
            'accuracy_score': accuracy_score,
            'context_chunks_used': len(context_chunks),
            'conversation_context_used': len(query_context.related_history) > 0,
            'retrieval_method': 'hybrid_with_conversation',
            'chunk_indices': expanded_indices
        }
        
        # Add to conversation memory
        self.conversation_memory.add_turn(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            confidence=final_confidence,
            chunk_indices=expanded_indices
        )
        
        # NO RESPONSE CACHING - removed cache operations
        
        return result
    
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
        
        # Final confidence
        confidence = avg_similarity + context_boost + type_boost
        return max(0.0, min(1.0, confidence))

# ========== MAIN APPLICATION ==========

# def main():
#     rag_system = ProductionRAGSystem()
    
#     # Load and process data
#     rag_system.load_and_process_data(DATA_PATH)
    
    # print(f"\n{'='*50}")
    # print("🚀 PRODUCTION RAG SYSTEM READY")
    # print("📈 Features: Enhanced Retrieval + Conversation Memory")
    # print("🧠 Cache: Embedding cache only (no response cache)")
    # print("🔍 Search: Hybrid semantic + BM25 + conversation context")
    # print(f"{'='*50}\n")
    
    # while True:
    #     user_query = input("💬 سوال پڇو (یا 'exit' چئو):\n> ").strip()
        
    #     if user_query.lower() in ['exit', 'quit', 'بند']:
    #         print("👋 خداحافظ!")
    #         break
        
    #     if not user_query:
    #         continue
        
    #     try:
    #         print("\n🔄 Processing...")
    #         result = rag_system.get_enhanced_response(user_query)
            
    #         print(f"\n🤖 جواب:\n{result['answer']}")
            
    #         # Low confidence warning
    #         if result['confidence'] < 0.3:
    #             print(f"\n⚠  نوٽ: هي جواب گهٽ اعتماد سان آهي ({result['confidence']:.2f})")
            
    #     except Exception as e:
    #         print(f"❌ Error: {str(e)}")
    #         logging.error(f"Query processing failed: {e}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     main()


# Global instance
_rag_system = None

def get_rag_system():
    """Get or create the RAG system instance"""
    global _rag_system
    if _rag_system is None:
        try:
            print("Initializing RAG system...")
            _rag_system = ProductionRAGSystem()
            # Load and process data during initialization
            _rag_system.load_and_process_data(DATA_PATH)
            print("RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            logger.error(f"RAG system initialization failed: {e}")
            raise e
    
    # Verify the system is properly initialized
    if _rag_system.faiss_index is None:
        print("FAISS index is None, reinitializing...")
        _rag_system.load_and_process_data(DATA_PATH)  # Add DATA_PATH here too
    
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
