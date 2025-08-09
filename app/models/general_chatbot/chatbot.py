import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from together import Together
from langchain.prompts import PromptTemplate
import hashlib
import pickle
from datetime import datetime, timedelta
from collections import deque
import json
import re
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from functools import lru_cache
from ...config.config import Config

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class ChatbotConfig:
    TOGETHER_API_KEY: str = getattr(Config, 'TOGETHER_API_KEY', '')
    MODEL_NAME: str = "BAAI/bge-m3"
    DATA_PATH: str = Path(getattr(Config, 'CHATBOT_DATA_PATH', 'app/models/general_chatbot/data/bhit_data.txt')).absolute()
    BHITTAIPEDIA_URL: str = "https://bhittaipedia.org/research/"
    INITIAL_RETRIEVE_K: int = 10
    RERANK_K: int = 4
    FINAL_CONTEXT_K: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.2
    WEB_MIN_SIMILARITY_THRESHOLD: float = 0.15
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 300
    MAX_CONVERSATION_HISTORY: int = 8
    WEB_CACHE_HOURS: int = 12
    MAX_WEB_PAGES: int = 2
    CACHE_DIR: Path = Path(__file__).parent / "cache"
    EMBEDDINGS_CACHE_DIR: Path = CACHE_DIR / "embeddings"
    WEB_EMBEDDINGS_CACHE_DIR: Path = CACHE_DIR / "web_embeddings"
    MODEL_CACHE_DIR: Path = CACHE_DIR / "embedding_model"

CONFIG = ChatbotConfig()

# Data Structures
@dataclass
class WebContent:
    url: str
    content: str
    scraped_at: datetime
    relevance_score: float
    content_hash: str

@dataclass
class ConversationTurn:
    query: str
    answer: str
    context_chunks: List[str]
    timestamp: datetime
    confidence: float
    chunk_indices: List[int]
    web_sources: List[str]

@dataclass
class QueryContext:
    query: str
    query_type: str
    conversation_context: str
    needs_web_search: bool

# Web Scraper
class WebScraper:
    def __init__(self):
        logger.info("Initializing WebScraper with cache directory: %s", CONFIG.CACHE_DIR)
        self.cache = {}
        self.cache_dir = CONFIG.CACHE_DIR / "web"
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        self._load_web_cache()
        logger.info("WebScraper initialized with %d cached items", len(self.cache))

    def _load_web_cache(self):
        logger.info("Loading web cache from: %s", self.cache_dir)
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with cache_file.open('rb') as f:
                    web_content = pickle.load(f)
                    if isinstance(web_content, WebContent) and datetime.now() - web_content.scraped_at < timedelta(hours=CONFIG.WEB_CACHE_HOURS):
                        cache_key = hashlib.md5(web_content.url.encode()).hexdigest()
                        self.cache[cache_key] = web_content
                        logger.info("Loaded cached web content for URL: %s", web_content.url)
                    else:
                        logger.info("Skipping outdated or invalid cache file: %s", cache_file)
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)
        logger.info("Web cache loaded, %d items", len(self.cache))

    def _get_cache_key(self, url: str) -> str:
        logger.info("Generating cache key for URL: %s", url)
        cache_key = hashlib.md5(url.encode()).hexdigest()
        logger.info("Generated cache key: %s", cache_key)
        return cache_key

    async def scrape(self, url: str, query: str) -> Optional[WebContent]:
        logger.info("Scraping URL: %s for query: %s", url, query)
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_key in self.cache and cache_file.exists():
            content = self.cache[cache_key]
            if datetime.now() - content.scraped_at < timedelta(hours=CONFIG.WEB_CACHE_HOURS):
                logger.info("Returning cached content for URL: %s with relevance: %.2f", url, content.relevance_score)
                return content
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10) as response:
                    response.raise_for_status()
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    content = ' '.join(p.get_text().strip() for p in soup.find_all(['p', 'article', 'div']) if p.get_text().strip())
                    if not content or len(content) < 100:
                        logger.info("No valid content found for URL: %s", url)
                        return None
                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                    relevance = self._calculate_relevance(content, query)
                    if relevance < CONFIG.WEB_MIN_SIMILARITY_THRESHOLD:
                        logger.info("Content relevance %.2f below threshold for URL: %s", relevance, url)
                        return None
                    web_content = WebContent(url, content[:2000], datetime.now(), relevance, content_hash)
                    self.cache[cache_key] = web_content
                    try:
                        with cache_file.open('wb') as f:
                            pickle.dump(web_content, f)
                    except PermissionError:
                        logger.warning("Permission denied when writing to cache file: %s", cache_file)
                    logger.info("Cached new content for URL: %s with relevance: %.2f", url, relevance)
                    return web_content
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: %s", e)
            return None

    def _calculate_relevance(self, content: str, query: str) -> float:
        logger.info("Calculating relevance for query: %s", query)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
        relevance = min(1.0, overlap + 0.1 if len(content) > 500 else 0)
        logger.info("Relevance score: %.2f for content length: %d", relevance, len(content))
        return relevance

# Conversation Memory
class ConversationMemory:
    def __init__(self):
        logger.info("Initializing ConversationMemory with max history: %d", CONFIG.MAX_CONVERSATION_HISTORY)
        self.history = deque(maxlen=CONFIG.MAX_CONVERSATION_HISTORY)
        logger.info("ConversationMemory initialized")

    def add_turn(self, query: str, answer: str, context_chunks: List[str], confidence: float, chunk_indices: List[int], web_sources: List[str]):
        logger.info("Adding conversation turn for query: %s", query)
        self.history.append(ConversationTurn(query, answer[:500], context_chunks[:3], datetime.now(), confidence, chunk_indices, web_sources))
        logger.info("Conversation turn added, history size: %d", len(self.history))

    @lru_cache(maxsize=100)
    def get_context(self, query: str) -> str:
        logger.info("Retrieving context for query: %s", query)
        if not self.history:
            logger.info("No conversation history available")
            return ""
        relevant = sorted(
            [turn for turn in self.history if any(word in turn.query.lower() for word in query.lower().split())],
            key=lambda t: (datetime.now() - t.timestamp).total_seconds(),
            reverse=True
        )[:2]
        context = "\n".join(f"Q: {t.query[:100]}\nA: {t.answer[:150]}" for t in relevant)
        logger.info("Retrieved context with %d relevant turns", len(relevant))
        return context

# RAG System
class ProductionRAGSystem:
    def __init__(self):
        logger.info("Initializing ProductionRAGSystem with model: %s", CONFIG.MODEL_NAME)
        CONFIG.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.MODEL_NAME, cache_dir=CONFIG.MODEL_CACHE_DIR, local_files_only=os.path.exists(CONFIG.MODEL_CACHE_DIR / CONFIG.MODEL_NAME)
        )
        self.model = AutoModel.from_pretrained(
            CONFIG.MODEL_NAME, cache_dir=CONFIG.MODEL_CACHE_DIR, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=os.path.exists(CONFIG.MODEL_CACHE_DIR / CONFIG.MODEL_NAME)
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.client = Together(api_key=CONFIG.TOGETHER_API_KEY)
        self.scraper = WebScraper()
        self.memory = ConversationMemory()
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.web_chunks = []
        self.web_embeddings = None
        self.web_faiss_index = None
        self.prompt = PromptTemplate.from_template("""
توھان شاھ عبداللطيف ڀٽائي جي زندگي، فڪر، ۽ ڪلام بابت جامع ۽ درست ڄاڻ رکندڙ ماھر آھيو، ۽ توھان جا سڀ جواب پاڪ-سنڌيءَ ۾، بغير اردو، انگريزي يا ٻي ٻولي جي ملاوٽ جي، پوري نفاست ۽ لسانياتي صحت سان لکيا ويندا.

سوال جي حوالي سان، مقامي حوالو (bhitt_data) کي 80 سيڪڙو ترجيح ڏيو ۽ ويب حوالو (bhittaipedia) کي صرف 20 سيڪڙو ترجيح ڏيو، جيستائين سوال ۾ خاص طور تي تازي معلومات جي ضرورت نه هجي.

{conversation_context}

حوالا (لازمي پيروي لاءِ):
1. جواب ۾ مقامي حوالو (bhitt_data) کي بنيادي طور استعمال ڪريو، ۽ ويب حوالو (bhittaipedia) صرف اضافي يا تازي معلومات لاءِ شامل ڪريو.
2. جڏھن توھان کان سوال پڇيو وڃي: "شاھ لطيف جي پُٽ جو نالو ڇا ھو؟" ته واضح جواب ڏيو: **"شاھ لطيف جو ڪو به پُٽ نه ھو"**.
3. شاھ لطيف جي زال جو نالو سعيدہ بيگم ھو، کيس ڪو به اولاد نه ھو.
4. جواب مڪمل، معلوماتي ۽ درست هجي، ۽ صرف ڏنل حوالن تي ٻڌل هجي.
5. جيڪڏھن سوال "شاھ لطيف ڪير ھو؟" آهي ته هيٺيون جواب استعمال ڪريو جيڪڏھن مقامي حوالو موجود هجي:
   **"شاھ عبداللطيف ڀٽائي (1689-1752) سنڌ جو عظيم صوفي شاعر ۽ فلسفي ھو، جنھن جو ڪلام 'شاھ جو رسالو' ۾ موجود آهي. سندس شاعري ۾ تصوف، انسانيت ۽ سنڌ جي ثقافت جا گہرا رنگ ملن ٿا. سندس زال سعيدہ بيگم ھئي، ۽ کين ڪو به اولاد نه ھو."**

موجوده سوال: {question}

مقامي حوالو:
{local_context}

جواب:
""")
        CONFIG.EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
        CONFIG.WEB_EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
        logger.info("ProductionRAGSystem initialized, model cached at: %s", CONFIG.MODEL_CACHE_DIR)

    async def load_data(self):
        logger.info("Loading data from path: %s", CONFIG.DATA_PATH)
        try:
            with open(CONFIG.DATA_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            data_hash = hashlib.sha256(text.encode()).hexdigest()
            embeddings_cache_file = CONFIG.EMBEDDINGS_CACHE_DIR / f"embeddings_{data_hash}.pkl"
            self.chunks = self._chunk_text(text)
            if embeddings_cache_file.exists():
                try:
                    with embeddings_cache_file.open('rb') as f:
                        cached = pickle.load(f)
                        if cached['hash'] == data_hash:
                            self.embeddings = cached['embeddings']
                            logger.info("Loaded cached embeddings for local data: %s", embeddings_cache_file)
                        else:
                            logger.info("Cached embeddings outdated, regenerating")
                            self.embeddings = self._generate_embeddings([c['text'] for c in self.chunks])
                            with embeddings_cache_file.open('wb') as f:
                                pickle.dump({'hash': data_hash, 'embeddings': self.embeddings}, f)
                except Exception as e:
                    logger.warning("Failed to load cached embeddings: %s, regenerating", e)
                    self.embeddings = self._generate_embeddings([c['text'] for c in self.chunks])
                    with embeddings_cache_file.open('wb') as f:
                        pickle.dump({'hash': data_hash, 'embeddings': self.embeddings}, f)
            else:
                self.embeddings = self._generate_embeddings([c['text'] for c in self.chunks])
                with embeddings_cache_file.open('wb') as f:
                    pickle.dump({'hash': data_hash, 'embeddings': self.embeddings}, f)
            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings)
            web_content = await self.scraper.scrape(CONFIG.BHITTAIPEDIA_URL, "")
            if web_content:
                self.web_chunks = self._chunk_text(web_content.content, source="web", url=web_content.url)
                web_embeddings_cache_file = CONFIG.WEB_EMBEDDINGS_CACHE_DIR / f"web_embeddings_{web_content.content_hash}.pkl"
                if web_embeddings_cache_file.exists():
                    try:
                        with web_embeddings_cache_file.open('rb') as f:
                            cached = pickle.load(f)
                            if cached['hash'] == web_content.content_hash:
                                self.web_embeddings = cached['embeddings']
                                logger.info("Loaded cached web embeddings: %s", web_embeddings_cache_file)
                            else:
                                logger.info("Cached web embeddings outdated, regenerating")
                                self.web_embeddings = self._generate_embeddings([c['text'] for c in self.web_chunks])
                                with web_embeddings_cache_file.open('wb') as f:
                                    pickle.dump({'hash': web_content.content_hash, 'embeddings': self.web_embeddings}, f)
                    except Exception as e:
                        logger.warning("Failed to load cached web embeddings: %s, regenerating", e)
                        self.web_embeddings = self._generate_embeddings([c['text'] for c in self.web_chunks])
                        with web_embeddings_cache_file.open('wb') as f:
                            pickle.dump({'hash': web_content.content_hash, 'embeddings': self.web_embeddings}, f)
                else:
                    self.web_embeddings = self._generate_embeddings([c['text'] for c in self.web_chunks])
                    with web_embeddings_cache_file.open('wb') as f:
                        pickle.dump({'hash': web_content.content_hash, 'embeddings': self.web_embeddings}, f)
                self.web_faiss_index = faiss.IndexFlatIP(self.web_embeddings.shape[1])
                faiss.normalize_L2(self.web_embeddings)
                self.web_faiss_index.add(self.web_embeddings)
            logger.info("Data loaded: %d local chunks, %d web chunks", len(self.chunks), len(self.web_chunks))
        except Exception as e:
            logger.error(f"Data loading failed: %s", e)
            raise

    def _chunk_text(self, text: str, source: str = "local", url: str = None) -> List[Dict]:
        logger.info("Chunking text from source: %s", source)
        sentences = re.split(r'[۔!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        start_idx = 0
        for i, sentence in enumerate(sentences):
            current_chunk += sentence + "۔ "
            if len(current_chunk.split()) >= CONFIG.CHUNK_SIZE:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': source,
                    'url': url,
                    'start_idx': start_idx,
                    'end_idx': i
                })
                overlap_sentences = sentences[max(start_idx, i - CONFIG.CHUNK_OVERLAP // 50):i]
                current_chunk = "۔ ".join(overlap_sentences) + "۔ "
                start_idx = max(0, i - CONFIG.CHUNK_OVERLAP // 50)
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source': source,
                'url': url,
                'start_idx': start_idx,
                'end_idx': len(sentences) - 1
            })
        logger.info("Created %d chunks from %s", len(chunks), source)
        return chunks

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        logger.info("Generating embeddings for %d texts", len(texts))
        if not texts:
            logger.info("No texts to embed, returning empty array")
            return np.array([])
        embeddings = []
        for i in range(0, len(texts), 32):
            inputs = self.tokenizer(texts[i:i+32], padding=True, truncation=True, return_tensors="pt", max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        logger.info("Generated %d embeddings", len(embeddings))
        return np.array(embeddings)

    def _classify_query(self, query: str) -> str:
        logger.info("Classifying query: %s", query)
        terms = {
            "birth": {"ڄميو", "پيدائش", "born", "birth"},
            "poetry": {"شاعري", "سُر", "رسالو", "poetry"},
            "family": {"زال", "پٽ", "wife", "son"},
            "philosophy": {"تصوف", "فلسفو", "sufism"}
        }
        query_lower = query.lower()
        for qtype, qterms in terms.items():
            if any(term in query_lower for term in qterms):
                logger.info("Query classified as: %s", qtype)
                return qtype
        logger.info("Query classified as: general")
        return "general"

    async def _retrieve(self, query: str, context: QueryContext) -> List[Dict]:
        logger.info("Retrieving chunks for query: %s, needs_web: %s", query, context.needs_web_search)
        query_embedding = self._generate_embeddings([query])
        results = []

        async def search_local():
            if self.faiss_index and self.faiss_index.ntotal > 0:
                scores, indices = self.faiss_index.search(query_embedding, CONFIG.INITIAL_RETRIEVE_K)
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.chunks):
                        similarity = float(score) * 1.4  # Strong boost for local
                        logger.debug("Local chunk %d: similarity=%.3f", idx, similarity)
                        if similarity > CONFIG.MIN_SIMILARITY_THRESHOLD:
                            results.append({'chunk': self.chunks[idx], 'score': similarity, 'source': 'local'})

        async def search_web():
            if context.needs_web_search or not results or max((r['score'] for r in results), default=0) < 0.4:
                web_content = await self.scraper.scrape(CONFIG.BHITTAIPEDIA_URL, query)
                if web_content and self.web_faiss_index and self.web_faiss_index.ntotal > 0:
                    web_scores, web_indices = self.web_faiss_index.search(query_embedding, CONFIG.INITIAL_RETRIEVE_K)
                    for score, idx in zip(web_scores[0], web_indices[0]):
                        if idx >= 0 and idx < len(self.web_chunks):
                            similarity = float(score) * 0.8  # Strong penalty for web
                            logger.debug("Web chunk %d: similarity=%.3f", idx, similarity)
                            if similarity > CONFIG.WEB_MIN_SIMILARITY_THRESHOLD:
                                results.append({'chunk': self.web_chunks[idx], 'score': similarity, 'source': 'web'})

        await asyncio.gather(search_local(), search_web())
        logger.info("Retrieved %d chunks (local: %d, web: %d)", len(results), sum(1 for r in results if r['source'] == 'local'), sum(1 for r in results if r['source'] == 'web'))
        return sorted(results, key=lambda x: x['score'], reverse=True)[:CONFIG.RERANK_K]

    async def get_response(self, query: str) -> Dict[str, Any]:
        logger.info("Generating response for query: %s", query)
        start_time = datetime.now()
        try:
            query_type = self._classify_query(query)
            history = self.memory.get_context(query)
            needs_web = "recent" in query.lower() or "latest" in query.lower()
            context = QueryContext(query, query_type, history, needs_web)
            results = await self._retrieve(query, context)
            if not results:
                logger.info("No relevant chunks found for query: %s", query)
                return {
                    'query': query,
                    'answer': "جواب نه مليو. مهرباني ڪري ٻيهر ڪوشش ڪريو.",
                    'confidence': 0.0,
                    'chunks_used': 0,
                    'web_sources': [],
                    'query_type': query_type
                }
            local_chunks = [r['chunk']['text'] for r in results if r['source'] == 'local'][:8]  # 80% of 10
            web_chunks = [r['chunk']['text'] for r in results if r['source'] == 'web'][:2]  # 20% of 10
            web_sources = list(set(r['chunk']['url'] for r in results if r['source'] == 'web' and r['chunk']['url']))
            confidence = min(0.95, 0.6 + 0.8 * (len(local_chunks) / 8) + 0.2 * (len(web_chunks) / 2))
            local_context = "\n".join(local_chunks) or "مقامي معلومات نه ملي"
            web_context = "\n".join(web_chunks) or "ويب معلومات نه ملي"
            source_info = f"مقامي چنڪ: {len(local_chunks)}, ويب چنڪ: {len(web_chunks)}"
            if query.strip() == "شاھ لطيف ڪير ھو؟" and local_context != "مقامي معلومات نه ملي":
                answer = "شاھ عبداللطيف ڀٽائي (1689-1752) سنڌ جو عظيم صوفي شاعر ۽ فلسفي ھو، جنھن جو ڪلام 'شاھ جو رسالو' ۾ موجود آهي. سندس شاعري ۾ تصوف، انسانيت ۽ سنڌ جي ثقافت جا گہرا رنگ ملن ٿا. سندس زال سعيدہ بيگم ھئي، ۽ کين ڪو به اولاد نه ھو."
                confidence = min(confidence + 0.2, 0.95)
            else:
                prompt = self.prompt.format(
                    question=query,
                    local_context=local_context,
                    web_context=web_context,
                    conversation_context=history,
                    confidence=confidence,
                    source_info=source_info
                )
                response = await asyncio.get_event_loop().run_in_executor(None, lambda: self.client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[{"role": "system", "content": "صرف سنڌي ۾ جواب ڏيو."}, {"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=800
                ))
                answer = response.choices[0].message.content.strip()
                if "پٽ" in query and "شاھ لطيف جو ڪو به پٽ نه ھو" not in answer:
                    answer = "شاھ لطيف جو ڪو به پٽ نه ھو"
                    confidence *= 0.5
            self.memory.add_turn(query, answer, local_chunks + web_chunks, confidence, [i for i, c in enumerate(self.chunks + self.web_chunks) if c['text'] in (local_chunks + web_chunks)], web_sources)
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info("Response generated with confidence: %.2f, chunks used: %d (local: %d, web: %d), time: %.2fs", 
                        confidence, len(local_chunks) + len(web_chunks), len(local_chunks), len(web_chunks), response_time)
            return {
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'chunks_used': len(local_chunks) + len(web_chunks),
                'web_sources': web_sources,
                'query_type': query_type
            }
        except Exception as e:
            logger.error(f"Response generation failed: %s", e)
            return {
                'query': query,
                'answer': f"خرابي: {str(e)}",
                'confidence': 0.0,
                'chunks_used': 0,
                'web_sources': [],
                'query_type': query_type
            }

# Manager
class RAGManager:
    def __init__(self):
        logger.info("Initializing RAGManager")
        self.rag = None
        self.lock = asyncio.Lock()
        logger.info("RAGManager initialized")

    async def get_rag(self):
        logger.info("Acquiring RAG system")
        async with self.lock:
            if self.rag is None:
                self.rag = ProductionRAGSystem()
                await self.rag.load_data()
            logger.info("RAG system acquired, initialized: %s", self.rag is not None)
            return self.rag

_manager = RAGManager()

def query_general_chatbot(query: str) -> str:
    """Main synchronous function for compatibility"""
    logger.info("Processing query in query_general_chatbot: %s", query)
    if not query or not query.strip():
        logger.info("Invalid query received, returning error message")
        return "مهرباني ڪري صحيح سوال ڏيو."
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, query_general_chatbot_async(query))
                result = future.result()
                logger.info("Query processed successfully, answer length: %d", len(result))
                return result
        except RuntimeError:
            result = asyncio.run(query_general_chatbot_async(query))
            logger.info("Query processed successfully, answer length: %d", len(result))
            return result
    except Exception as e:
        logger.error(f"Error in query: %s", e)
        return f"معذرت، خرابي آئي آهي: {str(e)}"

def query_general_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
    """Main synchronous session function for compatibility"""
    logger.info("Processing session query: %s, user_id: %s, session_id: %s", query, user_id, session_id)
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, query_general_chatbot_with_session_async(query, user_id, session_id))
                result = future.result()
                logger.info("Session query processed, answer length: %d", len(result.get('answer', '')))
                return result
        except RuntimeError:
            result = asyncio.run(query_general_chatbot_with_session_async(query, user_id, session_id))
            logger.info("Session query processed, answer length: %d", len(result.get('answer', '')))
            return result
    except Exception as e:
        logger.error(f"Error in session query: %s", e)
        return {
            'error': f"معذرت، خرابي آئي آهي: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }

async def query_general_chatbot_async(query: str) -> str:
    logger.info("Processing async query: %s", query)
    if not query or not query.strip():
        logger.info("Invalid async query received, returning error message")
        return "مهرباني ڪري صحيح سوال ڏيو."
    rag = await _manager.get_rag()
    result = await rag.get_response(query.strip())
    logger.info("Async query processed, answer length: %d", len(result['answer']))
    return result['answer']

async def query_general_chatbot_with_session_async(query: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
    from ...services.session_service import SessionService
    logger.info("Processing async session query: %s, user_id: %s, session_id: %s", query, user_id, session_id)
    if not query or not query.strip():
        logger.info("Invalid async session query received")
        return {'error': 'Query cannot be empty', 'code': 'INVALID_INPUT'}
    if not user_id:
        logger.info("Missing user_id in async session query")
        return {'error': 'User ID is required', 'code': 'INVALID_USER'}
    query = query.strip()
    if session_id:
        session = SessionService.get_session(session_id)
        if not session or not SessionService.verify_session_belongs_to_user(session_id, user_id):
            logger.info("Invalid session_id: %s for user: %s", session_id, user_id)
            return {'error': 'Invalid session', 'code': 'SESSION_INVALID'}
    else:
        session_id = SessionService.create_session(user_id, query[:50])
        logger.info("Created new session_id: %s for user: %s", session_id, user_id)
    user_message_id = SessionService.save_message(session_id, 'user', query)
    SessionService.update_session_activity(session_id)
    rag = await _manager.get_rag()
    start_time = datetime.now()
    result = await rag.get_response(query)
    response_time = (datetime.now() - start_time).total_seconds()
    bot_message_id = SessionService.save_message(session_id, 'bot', result['answer'])
    logger.info("Async session query processed, answer length: %d, confidence: %.2f", len(result['answer']), result['confidence'])
    total_chunks_used = result.get('chunks_used', 0)
    web_chunks_used = len(result.get('web_sources', []))
    local_chunks_used = total_chunks_used - web_chunks_used if total_chunks_used >= web_chunks_used else total_chunks_used
    return {
        'query': query,
        'answer': result['answer'],
        'session_id': session_id,
        'user_message_id': user_message_id,
        'bot_message_id': bot_message_id,
        'confidence': float(result['confidence']),
        'accuracy_score': min(1.0, float(result['confidence']) * 1.2),
        'context_chunks_used': total_chunks_used,
        'local_chunks_used': local_chunks_used,
        'web_chunks_used': web_chunks_used,
        'web_enhanced': web_chunks_used > 0,
        'web_sources': result['web_sources'],
        'query_type': result.get('query_type', 'general'),
        'response_time': response_time,
        'retrieval_method': 'hybrid' if web_chunks_used > 0 else 'local',
        'model': 'enhanced-web-rag-chatbot'
    }

def get_system_health() -> Dict[str, Any]:
    logger.info("Checking system health")
    health = {
        'system_initialized': _manager.rag is not None,
        'chunks_loaded': len(_manager.rag.chunks) if _manager.rag else 0,
        'web_chunks_loaded': len(_manager.rag.web_chunks) if _manager.rag else 0,
        'embeddings_cached': any(CONFIG.EMBEDDINGS_CACHE_DIR.glob("embeddings_*.pkl")) if _manager.rag else False,
        'web_embeddings_cached': any(CONFIG.WEB_EMBEDDINGS_CACHE_DIR.glob("web_embeddings_*.pkl")) if _manager.rag else False
    }
    logger.info("System health: initialized=%s, local_chunks=%d, web_chunks=%d, embeddings_cached=%s, web_embeddings_cached=%s", 
                health['system_initialized'], health['chunks_loaded'], health['web_chunks_loaded'], 
                health['embeddings_cached'], health['web_embeddings_cached'])
    return health

def get_rag_system():
    """Backward compatible synchronous function to get RAG system"""
    logger.info("Getting RAG system synchronously")
    try:
        try:
            loop = asyncio.get_running_loop()
            logger.warning("Already in async context, returning manager directly")
            logger.info("RAG system returned: %s", _manager)
            return _manager
        except RuntimeError:
            async def _get_rag():
                return await _manager.get_rag()
            rag = asyncio.run(_get_rag())
            logger.info("RAG system acquired synchronously: %s", rag)
            return rag
    except Exception as e:
        logger.error(f"Failed to get RAG system: %s", e)
        raise

def query_general_chatbot_sync(query: str) -> str:
    """Synchronous wrapper for query_general_chatbot"""
    logger.info("Processing sync query: %s", query)
    result = query_general_chatbot(query)
    logger.info("Sync query processed, answer length: %d", len(result))
    return result

def query_general_chatbot_with_session_sync(query: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for query_general_chatbot_with_session"""
    logger.info("Processing sync session query: %s, user_id: %s, session_id: %s", query, user_id, session_id)
    result = query_general_chatbot_with_session(query, user_id, session_id)
    logger.info("Sync session query processed, answer length: %d", len(result.get('answer', '')))
    return result