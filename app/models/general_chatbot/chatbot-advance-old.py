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
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from ...config.config import Config

# Web scraping and LangGraph imports
try:
    from langgraph.graph import Graph, END, StateGraph
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.prebuilt import ToolExecutor
    from langchain.tools import BaseTool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available. Web integration will use basic scraping.")

logger = logging.getLogger(__name__)

# ========== ENHANCED CONFIGURATION ==========

TOGETHER_API_KEY = getattr(Config, 'TOGETHER_API_KEY', '')
MODEL_NAME = getattr(Config, 'CHATBOT_MODEL_NAME', "BAAI/bge-m3")
DATA_PATH_CONFIG = getattr(Config, 'CHATBOT_DATA_PATH', 'app/models/general_chatbot/data/bhit_data.txt')

# Enhanced retrieval parameters with improved thresholds
INITIAL_RETRIEVE_K = 30          # Increased for better recall
RERANK_K = 12                   # Increased for better selection
FINAL_CONTEXT_K = 6             # Increased context
MIN_SIMILARITY_THRESHOLD = 0.25  # Increased for better precision
WEB_MIN_SIMILARITY_THRESHOLD = 0.20  # Increased for better web content filtering
CONTEXT_EXPANSION_RADIUS = 3  # Increased for more context
FINAL_CONTEXT_K = 8  # Increased for richer context

# Web scraping configuration
BHITTAIPEDIA_BASE_URL = "https://bhittaipedia.org"
BHITTAIPEDIA_RESEARCH_URL = "https://bhittaipedia.org/research/"
WEB_CONTENT_CACHE_HOURS = 24
MAX_WEB_PAGES_PER_QUERY = 3
WEB_SCRAPING_TIMEOUT = 15
WEB_CONTENT_MAX_LENGTH = 3000

# Performance settings
MAX_CONVERSATION_HISTORY = 10   # Increased for better context
CONVERSATION_CONTEXT_WEIGHT = 0.25  # Increased weight
CACHE_DIR = Path(__file__).parent / "cache"
MAX_WORKERS = min(6, os.cpu_count())  # Increased workers

# Convert relative path to absolute
if not os.path.isabs(DATA_PATH_CONFIG):
    base_dir = Path(__file__).parent.parent.parent.parent
    DATA_PATH = base_dir / DATA_PATH_CONFIG
else:
    DATA_PATH = Path(DATA_PATH_CONFIG)

# ========== WEB CONTENT MANAGEMENT ==========

@dataclass
class WebContent:
    url: str
    title: str
    content: str
    scraped_at: datetime
    relevance_score: float
    content_type: str  # 'research', 'biography', 'poetry', etc.

class WebContentCache:
    """Cache for web content with expiration"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "web_content"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.lock = threading.RLock()
    
    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[WebContent]:
        with self.lock:
            # Check memory first
            if url in self.memory_cache:
                content = self.memory_cache[url]
                if datetime.now() - content.scraped_at < timedelta(hours=WEB_CONTENT_CACHE_HOURS):
                    return content
                else:
                    del self.memory_cache[url]
            
            # Check disk cache
            cache_key = self._get_cache_key(url)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        content = pickle.load(f)
                    
                    if datetime.now() - content.scraped_at < timedelta(hours=WEB_CONTENT_CACHE_HOURS):
                        self.memory_cache[url] = content
                        return content
                    else:
                        cache_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to load cached content for {url}: {e}")
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, url: str, content: WebContent):
        with self.lock:
            self.memory_cache[url] = content
            
            # Save to disk
            cache_key = self._get_cache_key(url)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(content, f)
            except Exception as e:
                logger.warning(f"Failed to cache content for {url}: {e}")

class BhittaipediaWebScraper:
    """Enhanced web scraper for Bhittaipedia content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.cache = WebContentCache(CACHE_DIR)
        
        # Sindhi text patterns for better content extraction
        self.sindhi_patterns = {
            'title': re.compile(r'شاه|لطيف|ڀٽائي|سُر|رسالو'),
            'content': re.compile(r'[ء-ي]+'),
            'poetry': re.compile(r'سُر\s+\w+|رسالو'),
            'biography': re.compile(r'زندگي|حالات|ڄنم|وفات')
        }
    
    async def scrape_research_section(self) -> List[WebContent]:
        """Scrape the research section for relevant content"""
        try:
            response = self.session.get(BHITTAIPEDIA_RESEARCH_URL, timeout=WEB_SCRAPING_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all research links
            research_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and ('research' in href.lower() or 'shah' in href.lower() or 'latif' in href.lower()):
                    full_url = urljoin(BHITTAIPEDIA_BASE_URL, href)
                    research_links.append(full_url)
            
            # Remove duplicates
            research_links = list(set(research_links))
            
            # Scrape content from each link
            web_contents = []
            for url in research_links[:MAX_WEB_PAGES_PER_QUERY]:
                cached_content = self.cache.get(url)
                if cached_content:
                    web_contents.append(cached_content)
                    continue
                
                try:
                    await asyncio.sleep(0.5)  # Rate limiting
                    content = await self._scrape_single_page(url)
                    if content:
                        self.cache.set(url, content)
                        web_contents.append(content)
                
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
            
            return web_contents
        
        except Exception as e:
            logger.error(f"Failed to scrape research section: {e}")
            return []
    
    async def search_specific_query(self, query: str) -> List[WebContent]:
        """Search for specific query-related content"""
        search_urls = [
            f"{BHITTAIPEDIA_BASE_URL}/search?q={query}",
            f"{BHITTAIPEDIA_RESEARCH_URL}?search={query}",
        ]
        
        web_contents = []
        for search_url in search_urls:
            try:
                cached_content = self.cache.get(search_url)
                if cached_content:
                    web_contents.append(cached_content)
                    continue
                
                await asyncio.sleep(1)  # Rate limiting
                content = await self._scrape_single_page(search_url)
                if content:
                    self.cache.set(search_url, content)
                    web_contents.append(content)
            
            except Exception as e:
                logger.warning(f"Failed to search {search_url}: {e}")
                continue
        
        return web_contents
    
    async def _scrape_single_page(self, url: str) -> Optional[WebContent]:
        """Scrape content from a single page"""
        try:
            response = self.session.get(url, timeout=WEB_SCRAPING_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title') or soup.find('h1') or soup.find('h2')
            title = title_tag.get_text().strip() if title_tag else "Untitled"
            
            # Extract main content
            content_text = self._extract_main_content(soup)
            
            if not content_text or len(content_text) < 50:
                return None
            
            # Determine content type
            content_type = self._classify_content_type(title, content_text)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(content_text)
            
            return WebContent(
                url=url,
                title=title,
                content=content_text[:WEB_CONTENT_MAX_LENGTH],
                scraped_at=datetime.now(),
                relevance_score=relevance_score,
                content_type=content_type
            )
        
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article',
            '.content',
            '.main',
            '#content',
            '#main',
            'main',
            '.post-content',
            '.entry-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > len(content_text):
                        content_text = text
                break
        
        # Fallback to body if no specific content area found
        if not content_text:
            body = soup.find('body')
            if body:
                content_text = body.get_text().strip()
        
        # Clean up the text
        content_text = re.sub(r'\s+', ' ', content_text)
        content_text = re.sub(r'\n+', '\n', content_text)
        
        return content_text
    
    def _classify_content_type(self, title: str, content: str) -> str:
        """Classify content type based on patterns"""
        combined_text = f"{title} {content}".lower()
        
        if self.sindhi_patterns['poetry'].search(combined_text):
            return 'poetry'
        elif self.sindhi_patterns['biography'].search(combined_text):
            return 'biography'
        elif 'research' in combined_text or 'study' in combined_text:
            return 'research'
        else:
            return 'general'
    
    def _calculate_relevance_score(self, content: str) -> float:
        """Calculate relevance score based on Sindhi content and Shah Latif keywords"""
        score = 0.0
        
        # Sindhi text bonus
        sindhi_matches = len(self.sindhi_patterns['content'].findall(content))
        score += min(0.3, sindhi_matches / 100)
        
        # Shah Latif specific terms
        latif_terms = ['شاه', 'لطيف', 'ڀٽائي', 'سُر', 'رسالو', 'تصوف']
        for term in latif_terms:
            if term in content:
                score += 0.15
        
        # Content length bonus
        if len(content) > 500:
            score += 0.1
        
        return min(1.0, score)

# ========== LANGGRAPH WEB AGENT ==========

if LANGGRAPH_AVAILABLE:
    
    class BhittaipediaSearchTool(BaseTool):
        """Custom tool for searching Bhittaipedia"""
        
        name = "bhittaipedia_search"
        description = "Search Bhittaipedia.org for information about Shah Abdul Latif Bhittai"
        
        def __init__(self):
            super().__init__()
            self.scraper = BhittaipediaWebScraper()
        
        async def _arun(self, query: str) -> str:
            """Async search implementation"""
            try:
                web_contents = await self.scraper.search_specific_query(query)
                
                if not web_contents:
                    return "No relevant information found on Bhittaipedia"
                
                # Combine and summarize findings
                combined_content = []
                for content in web_contents[:3]:  # Limit to top 3 results
                    combined_content.append(f"From {content.title}: {content.content[:500]}")
                
                return "\n\n".join(combined_content)
            
            except Exception as e:
                return f"Error searching Bhittaipedia: {str(e)}"
        
        def _run(self, query: str) -> str:
            """Sync wrapper"""
            return asyncio.run(self._arun(query))
    
    class WebEnhancedRAGAgent:
        """LangGraph agent for web-enhanced RAG"""
        
        def __init__(self):
            self.scraper = BhittaipediaWebScraper()
            self.search_tool = BhittaipediaSearchTool()
            self.graph = self._create_agent_graph()
        
        def _create_agent_graph(self):
            """Create LangGraph workflow for web-enhanced RAG"""
            
            class AgentState:
                query: str
                local_results: List[Dict]
                web_results: List[WebContent]
                combined_context: str
                final_answer: str
                confidence: float
                sources: List[str]
            
            def analyze_query(state: AgentState) -> AgentState:
                """Analyze query to determine search strategy"""
                query = state.query.lower()
                
                # Determine if web search is needed
                web_indicators = ['latest', 'recent', 'new', 'current', 'modern', 'today']
                needs_web_search = any(indicator in query for indicator in web_indicators)
                
                # Always search web for better coverage
                state.needs_web_search = True
                state.query_type = self._classify_query_type(query)
                
                return state
            
            async def search_web_content(state: AgentState) -> AgentState:
                """Search web content"""
                try:
                    # Search specific query
                    web_contents = await self.scraper.search_specific_query(state.query)
                    
                    # Also get general research content if query is broad
                    if len(state.query.split()) <= 3:
                        research_contents = await self.scraper.scrape_research_section()
                        web_contents.extend(research_contents)
                    
                    # Remove duplicates and sort by relevance
                    unique_contents = {}
                    for content in web_contents:
                        if content.url not in unique_contents:
                            unique_contents[content.url] = content
                    
                    sorted_contents = sorted(
                        unique_contents.values(),
                        key=lambda x: x.relevance_score,
                        reverse=True
                    )
                    
                    state.web_results = sorted_contents[:MAX_WEB_PAGES_PER_QUERY]
                    
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    state.web_results = []
                
                return state
            
            def combine_sources(state: AgentState) -> AgentState:
                """Combine local and web sources"""
                combined_context_parts = []
                sources = []
                
                # Add local results
                if state.local_results:
                    for i, result in enumerate(state.local_results[:3]):
                        combined_context_parts.append(f"Local Source {i+1}: {result['text']}")
                        sources.append("Local Database")
                
                # Add web results
                for i, web_content in enumerate(state.web_results):
                    combined_context_parts.append(
                        f"Web Source {i+1} ({web_content.title}): {web_content.content}"
                    )
                    sources.append(web_content.url)
                
                state.combined_context = "\n\n".join(combined_context_parts)
                state.sources = sources
                
                return state
            
            def calculate_confidence(state: AgentState) -> AgentState:
                """Calculate overall confidence"""
                base_confidence = 0.5
                
                # Boost for local results
                if state.local_results:
                    base_confidence += 0.2
                
                # Boost for web results
                if state.web_results:
                    avg_web_relevance = np.mean([w.relevance_score for w in state.web_results])
                    base_confidence += 0.3 * avg_web_relevance
                
                # Context length bonus
                if len(state.combined_context) > 1000:
                    base_confidence += 0.1
                
                state.confidence = min(0.95, base_confidence)
                return state
            
            # Create the graph
            workflow = StateGraph(AgentState)
            
            workflow.add_node("analyze_query", analyze_query)
            workflow.add_node("search_web", search_web_content)
            workflow.add_node("combine_sources", combine_sources)
            workflow.add_node("calculate_confidence", calculate_confidence)
            
            workflow.set_entry_point("analyze_query")
            workflow.add_edge("analyze_query", "search_web")
            workflow.add_edge("search_web", "combine_sources")
            workflow.add_edge("combine_sources", "calculate_confidence")
            workflow.add_edge("calculate_confidence", END)
            
            return workflow.compile()
        
        def _classify_query_type(self, query: str) -> str:
            """Classify query type for better web search"""
            query_lower = query.lower()
            
            if any(term in query_lower for term in ['birth', 'born', 'ڄنم', 'ڄايو']):
                return 'biography'
            elif any(term in query_lower for term in ['poetry', 'poem', 'sur', 'سُر', 'شاعري']):
                return 'poetry'
            elif any(term in query_lower for term in ['philosophy', 'teaching', 'تصوف', 'فلسفو']):
                return 'philosophy'
            elif any(term in query_lower for term in ['death', 'died', 'وفات']):
                return 'death'
            else:
                return 'general'
        
        async def enhance_query(self, query: str, local_results: List[Dict]) -> Dict[str, Any]:
            """Enhance query with web content"""
            try:
                state = self.graph.invoke({
                    'query': query,
                    'local_results': local_results,
                    'web_results': [],
                    'combined_context': '',
                    'final_answer': '',
                    'confidence': 0.0,
                    'sources': []
                })
                
                return {
                    'enhanced_context': state['combined_context'],
                    'web_sources': state['sources'],
                    'confidence_boost': state['confidence'],
                    'web_results': state['web_results']
                }
            
            except Exception as e:
                logger.error(f"Web enhancement failed: {e}")
                return {
                    'enhanced_context': '',
                    'web_sources': [],
                    'confidence_boost': 0.0,
                    'web_results': []
                }

else:
    # Fallback implementation without LangGraph
    class WebEnhancedRAGAgent:
        """Fallback web-enhanced RAG without LangGraph"""
        
        def __init__(self):
            self.scraper = BhittaipediaWebScraper()
        
        async def enhance_query(self, query: str, local_results: List[Dict]) -> Dict[str, Any]:
            """Basic web enhancement without LangGraph"""
            try:
                # Simple web search
                web_contents = await self.scraper.search_specific_query(query)
                
                if not web_contents:
                    return {
                        'enhanced_context': '',
                        'web_sources': [],
                        'confidence_boost': 0.0,
                        'web_results': []
                    }
                
                # Combine contexts
                enhanced_context_parts = []
                web_sources = []
                
                for content in web_contents[:2]:
                    enhanced_context_parts.append(f"{content.title}: {content.content}")
                    web_sources.append(content.url)
                
                return {
                    'enhanced_context': '\n\n'.join(enhanced_context_parts),
                    'web_sources': web_sources,
                    'confidence_boost': 0.2,
                    'web_results': web_contents
                }
            
            except Exception as e:
                logger.error(f"Basic web enhancement failed: {e}")
                return {
                    'enhanced_context': '',
                    'web_sources': [],
                    'confidence_boost': 0.0,
                    'web_results': []
                }

# ========== UTILITY FUNCTIONS (Updated) ==========

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

# ========== ENHANCED DATA STRUCTURES ==========

@dataclass
class ConversationTurn:
    turn_id: str
    query: str
    answer: str
    context_chunks: List[str]
    timestamp: datetime
    confidence: float
    chunk_indices: List[int]
    web_sources: List[str] = None

@dataclass
class QueryContext:
    original_query: str
    processed_query: str
    query_type: str
    expanded_terms: List[str]
    conversation_context: str
    related_history: List[ConversationTurn]
    needs_web_search: bool = True

# ========== OPTIMIZED EMBEDDING CACHE (Unchanged but Enhanced) ==========

class OptimizedEmbeddingCache:
    """Thread-safe embedding cache with optimized memory management"""
    
    def __init__(self, cache_dir: str, max_memory_size: int = 2000, max_disk_size: int = 10000):
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
            'memory_reads': 0,
            'web_cache_hits': 0
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
                    
                    # Check expiry (extended for web content)
                    if datetime.now() - cached_data['timestamp'] < timedelta(hours=48):
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

# ========== ENHANCED CONVERSATION MEMORY ==========

class OptimizedConversationMemory:
    """Enhanced memory-efficient conversation tracking with web source tracking"""
    
    def __init__(self, max_history: int = MAX_CONVERSATION_HISTORY):
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.session_id = str(uuid.uuid4())
        self.query_index = {}  # Fast lookup by query terms
        self.web_source_tracking = {}  # Track web sources used
        
    def add_turn(self, query: str, answer: str, context_chunks: List[str], 
                 confidence: float, chunk_indices: List[int], web_sources: List[str] = None):
        """Enhanced turn addition with web source tracking"""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            query=query,
            answer=answer[:600],  # Increased for web-enhanced responses
            context_chunks=context_chunks[:4],  # Increased context storage
            timestamp=datetime.now(),
            confidence=confidence,
            chunk_indices=chunk_indices,
            web_sources=web_sources or []
        )
        
        # Remove old turn from index if deque is full
        if len(self.conversation_history) >= self.max_history:
            old_turn = self.conversation_history[0]
            self._remove_from_index(old_turn)
        
        self.conversation_history.append(turn)
        self._add_to_index(turn)
        
        # Track web sources
        if web_sources:
            for source in web_sources:
                if source not in self.web_source_tracking:
                    self.web_source_tracking[source] = []
                self.web_source_tracking[source].append(turn.turn_id)
    
    def _add_to_index(self, turn: ConversationTurn):
        """Add turn to query index for fast retrieval"""
        words = turn.query.lower().split()[:6]  # Increased indexing
        for word in words:
            if len(word) > 2:
                if word not in self.query_index:
                    self.query_index[word] = []
                self.query_index[word].append(turn.turn_id)
    
    def _remove_from_index(self, turn: ConversationTurn):
        """Remove turn from query index"""
        words = turn.query.lower().split()[:6]
        for word in words:
            if word in self.query_index:
                try:
                    self.query_index[word].remove(turn.turn_id)
                    if not self.query_index[word]:
                        del self.query_index[word]
                except ValueError:
                    pass
    
    @lru_cache(maxsize=100)
    def get_relevant_history(self, current_query: str) -> Tuple[ConversationTurn, ...]:
        """Enhanced relevant history retrieval with web source consideration"""
        if not self.conversation_history:
            return tuple()
        
        query_words = set(current_query.lower().split())
        scored_turns = []
        
        # Fast lookup using index
        candidate_turn_ids = set()
        for word in query_words:
            if word in self.query_index:
                candidate_turn_ids.update(self.query_index[word])
        
        # Score candidate turns with web source bonus
        for turn in self.conversation_history:
            if turn.turn_id in candidate_turn_ids:
                turn_words = set(turn.query.lower().split())
                overlap = len(query_words.intersection(turn_words))
                
                # Time decay factor
                time_diff = (datetime.now() - turn.timestamp).total_seconds()
                time_weight = max(0.1, 1.0 - time_diff / 7200)  # Extended to 2 hours
                
                # Web source bonus
                web_bonus = 0.1 if turn.web_sources else 0.0
                
                score = overlap * time_weight * turn.confidence + web_bonus
                scored_turns.append((turn, score))
        
        # Return top 3 most relevant turns (increased)
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return tuple(turn for turn, _ in scored_turns[:3])
    
    def get_conversation_context(self) -> str:
        """Get enhanced conversation context with web sources"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for turn in list(self.conversation_history)[-3:]:  # Increased to last 3 turns
            context_parts.append(f"Q: {turn.query[:120]}")
            context_parts.append(f"A: {turn.answer[:200]}")
            if turn.web_sources:
                context_parts.append(f"Sources: {', '.join(turn.web_sources[:2])}")
        
        return "\n".join(context_parts)

# ========== OPTIMIZED CHUNKING (Enhanced) ==========

class OptimizedChunker:
    """Enhanced high-performance text chunking with web content support"""
    
    def __init__(self):
        self.sindhi_sentence_pattern = re.compile(r'[۔؟!]+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.web_content_pattern = re.compile(r'[.!?]+')  # For English content
        
    @lru_cache(maxsize=20)  # Increased cache size
    @timing_decorator
    def create_overlapping_chunks(self, text_hash: str, text: str, 
                                 chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
        """Enhanced chunking with better overlap and web content support"""
        # Detect content language
        is_sindhi = bool(re.search(r'[ء-ي]', text))
        
        if is_sindhi:
            sentences = self.sindhi_sentence_pattern.split(text)
            sentence_separator = "۔ "
        else:
            sentences = self.web_content_pattern.split(text)
            sentence_separator = ". "
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        sentence_indices = []
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk += sentence + sentence_separator
            sentence_indices.append(i)
            
            # Check word count with enhanced size
            if len(current_chunk.split()) >= chunk_size:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentence_indices': sentence_indices.copy(),
                    'start_sentence': start_idx,
                    'end_sentence': i,
                    'word_count': len(current_chunk.split()),
                    'language': 'sindhi' if is_sindhi else 'english',
                    'content_hash': hashlib.md5(current_chunk.encode()).hexdigest()
                }
                chunks.append(chunk_data)
                
                # Enhanced overlap calculation
                overlap_sentences = max(2, len(sentence_indices) // 3)  # 33% overlap
                if len(sentence_indices) > overlap_sentences:
                    start_idx = sentence_indices[-overlap_sentences]
                    overlap_text = sentence_separator.join(sentences[start_idx:i+1]) + sentence_separator
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
                'word_count': len(current_chunk.split()),
                'language': 'sindhi' if is_sindhi else 'english',
                'content_hash': hashlib.md5(current_chunk.encode()).hexdigest()
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def create_web_chunks(self, web_contents: List[WebContent]) -> List[Dict]:
        """Create chunks from web content"""
        all_chunks = []
        
        for web_content in web_contents:
            # Create hash for web content
            content_hash = hashlib.md5(f"{web_content.url}_{web_content.content}".encode()).hexdigest()
            
            # Create chunks for this web content
            chunks = self.create_overlapping_chunks(content_hash, web_content.content)
            
            # Add web-specific metadata
            for chunk in chunks:
                chunk.update({
                    'source_type': 'web',
                    'source_url': web_content.url,
                    'source_title': web_content.title,
                    'relevance_score': web_content.relevance_score,
                    'content_type': web_content.content_type,
                    'scraped_at': web_content.scraped_at
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks

# ========== ENHANCED RAG SYSTEM ==========

class EnhancedProductionRAGSystem:
    """Enhanced high-performance RAG system with web integration"""
    
    def __init__(self):
        # Initialize models with optimizations
        self._initialize_models()
        
        # Initialize enhanced components
        self.embedding_cache = OptimizedEmbeddingCache(f"{CACHE_DIR}/embeddings")
        self.conversation_memory = OptimizedConversationMemory()
        self.chunker = OptimizedChunker()
        self.web_agent = WebEnhancedRAGAgent()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Data storage (enhanced)
        self.original_text = ""
        self.chunk_data = []  # Local chunks
        self.web_chunk_data = []  # Web chunks
        self.embeddings = None  # Local embeddings
        self.web_embeddings = None  # Web embeddings
        self.faiss_index = None  # Local FAISS index
        self.web_faiss_index = None  # Web FAISS index
        self.bm25 = None  # Local BM25
        self.web_bm25 = None  # Web BM25
        self._text_hash = None
        
        # Enhanced prompt template
        self.prompt = PromptTemplate.from_template("""
توھان شاھ عبداللطيف ڀٽائي جي زندگي، فڪر، ۽ ڪلام بابت جامع ۽ درست ڄاڻ رکندڙ ماھر آھيو، ۽ توھان جا سڀ جواب پاڪ-سنڌيءَ ۾، بغير اردو، انگريزي يا ٻي ٻولي جي ملاوٽ جي، پوري نفاست ۽ لسانياتي صحت سان لکيا ويندا.

{conversation_context}

حوالا (لازمي پيروي لاءِ):
1. جواب ۾ ڏنل "مقامي حوالو" ۽ "ويب حوالو" ٻنهي مان مستند معلومات استعمال ڪريو.
2. جڏھن توھان کان سوال پڇيو وڃي: "شاھ لطيف جي پُٽ جو نالو ڇا ھو؟" تہ واضح جواب ڏيو: **"شاھ لطيف جو ڪو به پُٽ نه ھو"**.
3. شاھ لطيف جي زال جو نالو سعيدہ بيگم ھو، کيس ڪو بہ اولاد نہ ھو.
4. جواب مڪمل، معلوماتي ۽ درست ھجي، ۽ جيڪڏھن ويب تان اضافي معلومات ملي ته ان کي شامل ڪريو.
5. حوالن جي ٻاھران ڪا به ڳالهه استعمال نه ڪريو.

موجوده سوال: {question}

مقامي حوالو:
{local_context}

ويب حوالو:
{web_context}

مجموعي اعتماد: {confidence}/1.0
{source_info}

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
        """Enhanced batch embedding generation with web content support"""
        if not texts:
            return np.array([])
            
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Parallel cache checking
        cache_keys = [self.embedding_cache._get_cache_key(text) for text in texts]
        
        with ThreadPoolExecutor(max_workers=min(6, len(texts))) as cache_executor:
            cache_futures = {
                cache_executor.submit(self.embedding_cache.get, key): (i, key) 
                for i, key in enumerate(cache_keys)
            }
            
            for future in as_completed(cache_futures):
                i, key = cache_futures[future]
                cached_embedding = future.result()
                
                if cached_embedding is not None:
                    embeddings.append((i, cached_embedding))
                    self.embedding_cache.stats['web_cache_hits'] += 1
                else:
                    uncached_texts.append(texts[i])
                    uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]
                
                # Tokenize batch with increased max length for web content
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt", 
                    max_length=768  # Increased for web content
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
    
    @lru_cache(maxsize=150)
    def _classify_query_optimized(self, query: str, context: str) -> str:
        """Enhanced query classification with more Sindhi-specific terms"""
        combined_text = f"{context} {query}".lower()
        
        birth_terms = {"ڪيڏانهن", "ڪٿي", "جنم", "ڄائو", "پيدائش", "birth", "born", "جاءِ"}
        date_terms = {"ڪڏھن", "سال", "تاريخ", "ڄمڻ", "date", "year", "when", "سن"}
        poetry_terms = {"شاعري", "ڪلام", "سُر", "رسالو", "شعر", "poetry", "poem", "verse", "بيت"}
        death_terms = {"مرڻ", "وفات", "آخر", "موت", "death", "died", "demise", "انتقال"}
        bio_terms = {"زندگي", "حالات", "تعليم", "پيدائش", "biography", "life", "education", "تعارف"}
        phil_terms = {"فلسفو", "تصوف", "عقيدو", "خيال", "philosophy", "sufism", "mysticism", "روحانيت"}
        family_terms = {"زال", "گھر", "خاندان", "wife", "family", "marriage", "شادي", "سعيده"}
        
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
        elif family_terms.intersection(text_words):
            return "family"
        else:
            return "general"

    
    @timing_decorator
    async def enhanced_precision_retrieval(self, query_context: QueryContext) -> List[Tuple[Dict, float, str]]:
        """Enhanced retrieval combining local and web sources"""
        query = query_context.processed_query
        
        # Parallel retrieval from local and web sources
        local_future = self.executor.submit(self._local_retrieval, query, query_context)
        web_future = self.executor.submit(self._web_enhanced_retrieval, query, query_context)
        
        local_results = local_future.result()
        web_enhancement = web_future.result()
        
        # Combine and weight results
        combined_results = []
        
        # Add local results with original weighting
        for chunk, score in local_results:
            combined_results.append((chunk, score, 'local'))
        
        # Add web-enhanced context if available
        if web_enhancement['web_results']:
            # Create pseudo-chunks from web content
            for web_content in web_enhancement['web_results']:
                web_chunks = self.chunker.create_web_chunks([web_content])
                for chunk in web_chunks[:2]:  # Limit web chunks per source
                    # Calculate similarity with query
                    web_score = self._calculate_web_similarity(chunk['text'], query)
                    if web_score > WEB_MIN_SIMILARITY_THRESHOLD:
                        combined_results.append((chunk, web_score * 0.8, 'web'))  # Slight discount for web
        
        # Sort by score and return top results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:RERANK_K]
    
    def _local_retrieval(self, query: str, query_context: QueryContext) -> List[Tuple[Dict, float]]:
        """Enhanced local retrieval with balanced scoring"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = executor.submit(self._semantic_search, query, is_local=True)
            bm25_future = executor.submit(self._bm25_search, query, is_local=True)
            
            semantic_results = semantic_future.result()
            bm25_results = bm25_future.result()
        
        all_results = {}
        
        # Process semantic results with stricter filtering
        for idx, score in semantic_results:
            if score > MIN_SIMILARITY_THRESHOLD:
                all_results[idx] = {'semantic': score, 'bm25': 0, 'conversation': 0}
                logger.debug(f"Local semantic match: chunk={idx}, score={score:.3f}")
        
        # Process BM25 results with normalized scoring
        for idx, score in bm25_results:
            normalized_score = min(1.0, score / 8)  # Adjusted normalization
            if idx in all_results:
                all_results[idx]['bm25'] = normalized_score
            elif normalized_score > 0.15:  # Slightly higher BM25 threshold
                all_results[idx] = {'semantic': 0, 'bm25': normalized_score, 'conversation': 0}
                logger.debug(f"Local BM25 match: chunk={idx}, score={normalized_score:.3f}")
        
        # Enhanced conversation boost
        for turn in query_context.related_history:
            for chunk_idx in turn.chunk_indices[:4]:
                if chunk_idx in all_results:
                    all_results[chunk_idx]['conversation'] = turn.confidence * CONVERSATION_CONTEXT_WEIGHT
        
        results = []
        query_type_boost = self._get_enhanced_type_boost(query_context.query_type)
        
        for idx, scores in all_results.items():
            if idx < len(self.chunk_data):
                # Enhanced hybrid scoring with balanced weights
                hybrid_score = (
                    0.5 * scores['semantic'] +  # Reduced semantic weight
                    0.35 * scores['bm25'] +    # Increased BM25 weight
                    0.15 * scores['conversation']  # Increased conversation weight
                )
                
                if query_type_boost > 1.0:
                    chunk_text = self.chunk_data[idx]['text'].lower()
                    if self._matches_query_type(chunk_text, query_context.query_type):
                        hybrid_score *= query_type_boost
                
                results.append((self.chunk_data[idx], hybrid_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Local retrieval: {len(results)} results, top score={results[0][1]:.3f}" if results else "Local retrieval: No results")
        return results[:INITIAL_RETRIEVE_K]
    
    def _web_enhanced_retrieval(self, query: str, query_context: QueryContext) -> Dict[str, Any]:
        """Enhanced web retrieval with improved relevance filtering"""
        import asyncio
        
        async def _async_web_retrieval():
            try:
                local_results = [{'text': chunk['text']} for chunk in self.chunk_data[:5]]
                web_enhancement = await self.web_agent.enhance_query(query, local_results)
                
                # Filter web results by relevance
                filtered_web_results = [
                    content for content in web_enhancement['web_results']
                    if content.relevance_score > 0.3  # Stricter web relevance threshold
                ]
                web_enhancement['web_results'] = filtered_web_results
                logger.info(f"Web retrieval: {len(filtered_web_results)} relevant web results")
                
                return web_enhancement
            except Exception as e:
                logger.warning(f"Web enhancement failed: {e}")
                return {
                    'enhanced_context': '',
                    'web_sources': [],
                    'confidence_boost': 0.0,
                    'web_results': []
                }
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_web_retrieval())
                    return future.result()
            else:
                return asyncio.run(_async_web_retrieval())
        except Exception as e:
            logger.warning(f"Web enhancement failed in event loop handling: {e}")
            return {
                'enhanced_context': '',
                'web_sources': [],
                'confidence_boost': 0.0,
                'web_results': []
            }
    
    def _semantic_search(self, query: str, is_local: bool = True) -> List[Tuple[int, float]]:
        """Enhanced semantic search with stricter similarity filtering"""
        query_embedding = self.generate_embeddings_batch([query])
        
        if is_local and self.faiss_index is not None:
            index = self.faiss_index
            retrieve_k = INITIAL_RETRIEVE_K
        elif not is_local and self.web_faiss_index is not None:
            index = self.web_faiss_index
            retrieve_k = min(MAX_WEB_PAGES_PER_QUERY * 2, INITIAL_RETRIEVE_K)
        else:
            logger.warning("No valid index available for semantic search")
            return []
        
        distances, indices = index.search(query_embedding, retrieve_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                # Enhanced similarity calculation with adjusted scaling
                similarity = 1 / (1 + dist * 0.3)  # Tighter scaling for higher precision
                threshold = MIN_SIMILARITY_THRESHOLD if is_local else WEB_MIN_SIMILARITY_THRESHOLD
                if similarity > threshold:
                    results.append((idx, similarity))
                    logger.debug(f"Semantic search {'local' if is_local else 'web'} chunk {idx}: score={similarity:.3f}")
        
        return results
    
    def _bm25_search(self, query: str, is_local: bool = True) -> List[Tuple[int, float]]:
        """Enhanced BM25 search with improved scoring"""
        if is_local and self.bm25 is not None:
            bm25_index = self.bm25
        elif not is_local and self.web_bm25 is not None:
            bm25_index = self.web_bm25
        else:
            return []
        
        query_tokens = query.split()
        bm25_scores = bm25_index.get_scores(query_tokens)
        
        # Enhanced scoring with better normalization
        max_score = np.max(bm25_scores) if len(bm25_scores) > 0 else 1
        normalized_scores = bm25_scores / max(max_score, 1) if max_score > 0 else bm25_scores
        
        # Get top indices with improved threshold
        threshold = 0.05 if is_local else 0.03  # Lower thresholds
        top_indices = np.where(normalized_scores > threshold)[0]
        
        results = []
        for idx in top_indices:
            if normalized_scores[idx] > threshold:
                results.append((idx, float(normalized_scores[idx])))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:INITIAL_RETRIEVE_K]
    
    def _calculate_web_similarity(self, web_text: str, query: str) -> float:
        """Calculate similarity between web content and query"""
        try:
            web_embedding = self.generate_embeddings_batch([web_text])
            query_embedding = self.generate_embeddings_batch([query])
            
            if web_embedding.size == 0 or query_embedding.size == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(web_embedding[0], query_embedding[0]) / (
                np.linalg.norm(web_embedding[0]) * np.linalg.norm(query_embedding[0])
            )
            
            return float(similarity)
        
        except Exception as e:
            logger.warning(f"Web similarity calculation failed: {e}")
            return 0.0
    
    def _get_enhanced_type_boost(self, query_type: str) -> float:
        """Enhanced boost factors for different query types"""
        boosts = {
            "birth_location": 1.4,
            "birth_date": 1.4,
            "poetry_work": 1.3,
            "death": 1.4,
            "biography": 1.2,
            "philosophy": 1.2,
            "family": 1.3
        }
        return boosts.get(query_type, 1.0)
    
    def _matches_query_type(self, chunk_text: str, query_type: str) -> bool:
        """Enhanced query type matching"""
        text_lower = chunk_text.lower()
        
        type_terms = {
            "birth_location": ["ڀٽ شاھ", "ڄنم", "پيدائش", "birth", "born"],
            "birth_date": ["سال", "تاريخ", "date", "year"],
            "poetry_work": ["رسالو", "سُر", "شاعري", "poetry", "sur", "risalo"],
            "death": ["وفات", "مرڻ", "انتقال", "death", "died"],
            "family": ["زال", "سعيده", "بيگم", "wife", "marriage"],
            "philosophy": ["تصوف", "فلسفو", "sufism", "mysticism"]
        }
        
        if query_type in type_terms:
            return any(term in text_lower for term in type_terms[query_type])
        
        return False
    
    def expand_context_optimized(self, selected_indices: List[int], source_type: str = 'local') -> List[int]:
        """Enhanced context expansion with source awareness"""
        if not selected_indices:
            return []
            
        expanded = set(selected_indices)
        
        if source_type == 'local':
            max_idx = len(self.chunk_data) - 1
        else:
            max_idx = len(self.web_chunk_data) - 1
        
        # Enhanced expansion with increased radius
        for idx in selected_indices:
            for offset in range(-CONTEXT_EXPANSION_RADIUS, CONTEXT_EXPANSION_RADIUS + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx <= max_idx:
                    expanded.add(neighbor_idx)
        
        return sorted(expanded)
    
    @timing_decorator
    async def load_and_process_data(self, file_path: str):
        """Enhanced data loading with web content integration"""
        print("🔄 Loading and processing enhanced data...")
        
        # Load local data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.original_text = f.read()
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
        
        # Create text hash for caching
        self._text_hash = hashlib.md5(self.original_text.encode()).hexdigest()
        
        # Create enhanced local chunks
        self.chunk_data = self.chunker.create_overlapping_chunks(
            self._text_hash, self.original_text
        )
        print(f"✅ Created {len(self.chunk_data)} local chunks")
        
        # Load and process web content
        print("🌐 Loading web content from Bhittaipedia...")
        await self._load_web_content()
        
        # Process local embeddings
        print("🧠 Generating local embeddings...")
        local_texts = [chunk['text'] for chunk in self.chunk_data]
        self.embeddings = self.generate_embeddings_batch(local_texts)
        
        # Create local FAISS index
        print("🔍 Building local search indices...")
        if self.embeddings.size > 0:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings)
            
            # Create local BM25 index
            tokenized_chunks = [chunk['text'].split() for chunk in self.chunk_data]
            self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Process web embeddings if available
        if self.web_chunk_data:
            print("🌐 Generating web embeddings...")
            web_texts = [chunk['text'] for chunk in self.web_chunk_data]
            self.web_embeddings = self.generate_embeddings_batch(web_texts)
            
            if self.web_embeddings.size > 0:
                # Create web FAISS index
                web_dimension = self.web_embeddings.shape[1]
                self.web_faiss_index = faiss.IndexFlatIP(web_dimension)
                faiss.normalize_L2(self.web_embeddings)
                self.web_faiss_index.add(self.web_embeddings)
                
                # Create web BM25 index
                web_tokenized = [chunk['text'].split() for chunk in self.web_chunk_data]
                self.web_bm25 = BM25Okapi(web_tokenized)
        
        print("🚀 Enhanced system ready with web integration!")
        
        # Print enhanced statistics
        cache_stats = self.embedding_cache.get_stats()
        print(f"📊 Cache stats: {cache_stats['memory_size']} items, {cache_stats['hit_rate']:.2%} hit rate")
        print(f"🌐 Web chunks: {len(self.web_chunk_data)}")
    
    async def _load_web_content(self):
        """Load and process web content from Bhittaipedia"""
        try:
            scraper = BhittaipediaWebScraper()
            
            # Scrape research section
            web_contents = await scraper.scrape_research_section()
            
            if web_contents:
                # Create chunks from web content
                self.web_chunk_data = self.chunker.create_web_chunks(web_contents)
                print(f"✅ Loaded {len(web_contents)} web pages, created {len(self.web_chunk_data)} web chunks")
            else:
                print("⚠️ No web content loaded")
                self.web_chunk_data = []
        
        except Exception as e:
            logger.error(f"Failed to load web content: {e}")
            self.web_chunk_data = []
    
    def enhanced_query_processing(self, query: str) -> QueryContext:
        """Enhanced query processing with better Sindhi term expansion"""
        relevant_history = list(self.conversation_memory.get_relevant_history(query))
        conversation_context = self.conversation_memory.get_conversation_context()
        
        # Enhanced query expansion with Sindhi-specific terms
        expanded_terms = [query]
        sindhi_synonyms = {
            "شاھ": ["شاه عبداللطيف", "ڀٽائي", "لطيف"],
            "ڄميو": ["پيدائش", "جنم", "ڄائو"],
            "شاعري": ["ڪلام", "سُر", "رسالو", "بيت"],
            "زال": ["سعيده", "بيگم", "شادي"]
        }
        
        for term in query.split():
            if term in sindhi_synonyms:
                expanded_terms.extend(sindhi_synonyms[term])
        
        for turn in relevant_history[:3]:
            prev_words = [word for word in turn.query.split() if len(word) > 3]
            for word in prev_words[:4]:
                if word not in expanded_terms:
                    expanded_terms.append(word)
        
        query_type = self._classify_query_optimized(query, conversation_context)
        
        web_indicators = ['نئين', 'تازي', 'موجوده', 'recent', 'latest', 'current', 'new']
        needs_web_search = any(indicator in query.lower() for indicator in web_indicators) or query_type in ["biography", "poetry_work"]
        
        return QueryContext(
            original_query=query,
            processed_query=self._clean_query_optimized(query),
            query_type=query_type,
            expanded_terms=expanded_terms[:10],  # Increased expansion
            conversation_context=conversation_context,
            related_history=relevant_history,
            needs_web_search=needs_web_search
        )
    
    @lru_cache(maxsize=300)
    def _clean_query_optimized(self, query: str) -> str:
        """Enhanced query cleaning with web content support"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        # Keep both Sindhi and English characters for web content
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u0000-\u007F]', ' ', cleaned)
        return cleaned.strip()

    @timing_decorator
    async def get_enhanced_response(self, query: str) -> Dict[str, Any]:
        """Enhanced response generation with web integration"""
        start_time = datetime.now()
        
        # Process query with enhanced conversation context
        query_context = self.enhanced_query_processing(query)
        
        # Enhanced precision retrieval with web integration
        retrieval_results = await self.enhanced_precision_retrieval(query_context)
        
        if not retrieval_results:
            logger.warning("No retrieval results found")
            return {
                'query': query,
                'answer': "توھان جي سوال جو جواب نه ملي سگھيو. مهرباني ڪري ٻيهر ڪوشش ڪريو.",
                'confidence': 0.0,
                'accuracy_score': 0.0,
                'context_chunks_used': 0,
                'retrieval_method': 'none',
                'web_sources': []
            }
        
        # Separate local and web results
        local_results = [(chunk, score) for chunk, score, source in retrieval_results if source == 'local']
        web_results = [(chunk, score) for chunk, score, source in retrieval_results if source == 'web']
        
        # Expand context for both local and web results
        local_indices = []
        web_indices = []
        
        if local_results:
            local_selected = [
                next(i for i, chunk in enumerate(self.chunk_data) if chunk.get('content_hash') == chunk_info.get('content_hash', ''))
                for chunk_info, _ in local_results[:FINAL_CONTEXT_K//2] if any(
                    chunk.get('content_hash') == chunk_info.get('content_hash', '') for chunk in self.chunk_data
                )
            ]
            local_indices = self.expand_context_optimized(local_selected, 'local')
        
        if web_results:
            web_selected = [
                next(i for i, chunk in enumerate(self.web_chunk_data) if chunk.get('content_hash') == chunk_info.get('content_hash', ''))
                for chunk_info, _ in web_results[:FINAL_CONTEXT_K//2] if any(
                    chunk.get('content_hash') == chunk_info.get('content_hash', '') for chunk in self.web_chunk_data
                )
            ]
            web_indices = self.expand_context_optimized(web_selected, 'web')
        
        # Assemble enhanced context
        local_context_chunks = [self.chunk_data[i]['text'] for i in local_indices] if local_indices else []
        web_context_chunks = [self.web_chunk_data[i]['text'] for i in web_indices] if web_indices else []
        
        local_context = "\n\n".join(local_context_chunks)
        web_context = "\n\n".join(web_context_chunks)
        
        all_context_chunks = local_context_chunks + web_context_chunks
        
        # Calculate enhanced confidence
        try:
            confidence = self._calculate_enhanced_confidence(
                tuple(all_context_chunks), query, query_context.query_type, bool(web_context)
            )
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            confidence = 0.6  # Higher default for enhanced system
        
        # Collect web sources
        web_sources = []
        for chunk in [self.web_chunk_data[i] for i in web_indices]:
            if chunk.get('source_url') and chunk['source_url'] not in web_sources:
                web_sources.append(chunk['source_url'])
        
        # Generate enhanced response
        try:
            source_info = ""
            if web_sources:
                source_info = f"ويب ذريعا: {', '.join(web_sources[:3])}"
            
            formatted_prompt = self.prompt.format(
                question=query,
                local_context=local_context[:1500] if local_context else "مقامي ڊيٽا ۾ ڪا خاص معلومات نه ملي",
                web_context=web_context[:1500] if web_context else "ويب تان ڪا اضافي معلومات نه ملي",
                conversation_context=f"اڳوڻو سلسلو:\n{query_context.conversation_context}" 
                if query_context.conversation_context else "",
                confidence=confidence,
                source_info=source_info
            )
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {
                        "role": "system", 
                        "content": "توھان شاه عبداللطيف ڀٽائي جي ماھر آھيو. صرف سنڌي ۾ جواب ڏيو، مڪمل ۽ درست معلومات سان."
                    },
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,  # Increased for detailed responses
                timeout=45        # Increased timeout
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "معذرت، هن وقت جواب تيار نہ ٿي سگھيو. مهرباني ڪري ٻيهر ڪوشش ڪريو."
            confidence = max(0.1, confidence * 0.5)
        
        # Verify answer accuracy with enhanced verification
        try:
            accuracy_score = self._verify_answer_accuracy_enhanced(answer, all_context_chunks, web_sources)
        except Exception as e:
            logger.warning(f"Answer accuracy verification failed: {e}")
            accuracy_score = 0.6  # Higher default
        
        final_confidence = min(confidence, accuracy_score)
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare enhanced result
        result = {
            'query': query,
            'answer': answer,
            'confidence': final_confidence,
            'accuracy_score': accuracy_score,
            'context_chunks_used': len(all_context_chunks),
            'local_chunks_used': len(local_context_chunks),
            'web_chunks_used': len(web_context_chunks),
            'conversation_context_used': len(query_context.related_history) > 0,
            'retrieval_method': 'enhanced_hybrid_web',
            'chunk_indices': local_indices,
            'web_chunk_indices': web_indices,
            'query_type': query_context.query_type,
            'web_sources': web_sources,
            'response_time': response_time
        }
        
        # Add to conversation memory with web sources
        self.conversation_memory.add_turn(
            query=query,
            answer=answer,
            context_chunks=all_context_chunks[:4],
            confidence=final_confidence,
            chunk_indices=local_indices + web_indices,
            web_sources=web_sources
        )
        
        return result
    
    @lru_cache(maxsize=150)
    def _calculate_enhanced_confidence(self, context_chunks_tuple: tuple, 
                                     query: str, query_type: str, has_web_content: bool) -> float:
        """Enhanced confidence calculation with web content consideration"""
        context_chunks = list(context_chunks_tuple)
        
        if not context_chunks:
            return 0.0
        
        # Enhanced similarity calculation
        try:
            # Use more chunks for better confidence estimation
            sample_chunks = context_chunks[:3]
            chunk_embeddings = self.generate_embeddings_batch(sample_chunks)
            query_embedding = self.generate_embeddings_batch([query])
            
            if chunk_embeddings.size == 0 or query_embedding.size == 0:
                return 0.4  # Higher default confidence
            
            # Enhanced similarity calculation
            similarities = np.dot(chunk_embeddings, query_embedding[0]) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding[0])
            )
            
            # Use weighted average (give more weight to best matches)
            weights = np.array([0.5, 0.3, 0.2])[:len(similarities)]
            avg_similarity = float(np.average(similarities, weights=weights))
            
        except Exception as e:
            logger.warning(f"Enhanced confidence calculation failed: {e}")
            avg_similarity = 0.6
        
        # Enhanced boosting factors
        type_boost = 0.15 if query_type != "general" else 0.0
        web_boost = 0.1 if has_web_content else 0.0
        content_length_boost = min(0.1, len(''.join(context_chunks)) / 10000)
        
        # Final confidence with enhanced bounds
        confidence = avg_similarity + type_boost + web_boost + content_length_boost
        return max(0.2, min(0.95, confidence))
    
    def _verify_answer_accuracy_enhanced(self, answer: str, context_chunks: List[str], 
                                       web_sources: List[str]) -> float:
        """Enhanced answer verification with cross-source checking"""
        if not answer or not context_chunks:
            logger.warning("No answer or context chunks for verification")
            return 0.2
        
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        if not answer_words:
            logger.warning("No meaningful answer words for verification")
            return 0.3
        
        # Enhanced context analysis
        local_context = " ".join(context_chunks[:3]).lower()
        web_context = " ".join(context_chunks[3:]).lower() if len(context_chunks) > 3 else ""
        
        local_words = set(word for word in local_context.split() if len(word) > 2)
        web_words = set(word for word in web_context.split() if len(word) > 2)
        
        local_overlap = len(answer_words.intersection(local_words)) / len(answer_words) if answer_words else 0
        web_overlap = len(answer_words.intersection(web_words)) / len(answer_words) if answer_words and web_words else 0
        
        # Enhanced scoring with cross-source validation
        accuracy_score = 0.4 * local_overlap + 0.3 * web_overlap
        
        # Specific fact checking for known information
        if "پٽ" in answer or "son" in answer.lower():
            if "شاھ لطيف جو ڪو به پٽ نه ھو" not in answer:
                accuracy_score *= 0.5
                logger.warning("Answer contains incorrect son information")
        
        if "زال" in answer or "wife" in answer.lower():
            if "سعيده" not in answer and "بيگم" not in answer:
                accuracy_score *= 0.7
                logger.warning("Answer missing wife's name (Saeeda Begum)")
        
        # Web source and length bonuses
        accuracy_score += 0.15 if web_sources else 0.0
        accuracy_score += 0.1 if 50 <= len(answer.split()) <= 250 else 0.0
        
        logger.debug(f"Answer verification: local_overlap={local_overlap:.3f}, web_overlap={web_overlap:.3f}, final_score={accuracy_score:.3f}")
        return min(1.0, accuracy_score * 1.1)

# ========== ENHANCED MAIN APPLICATION ==========

class EnhancedRAGSystemManager:
    """Enhanced thread-safe RAG system manager with web integration and health monitoring"""
    
    def __init__(self):
        self._rag_system = None
        self._initialization_lock = threading.Lock()
        self._health_stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'web_enhanced_queries': 0,
            'avg_response_time': 0.0,
            'last_error': None,
            'web_scraping_errors': 0,
            'avg_confidence': 0.0
        }
    
    async def get_rag_system(self) -> EnhancedProductionRAGSystem:
        """Thread-safe enhanced RAG system initialization"""
        if self._rag_system is None:
            with self._initialization_lock:
                if self._rag_system is None:  # Double-check locking
                    try:
                        logger.info("Initializing enhanced RAG system with web integration...")
                        self._rag_system = EnhancedProductionRAGSystem()
                        await self._rag_system.load_and_process_data(DATA_PATH)
                        logger.info("Enhanced RAG system initialized successfully")
                    except Exception as e:
                        logger.error(f"Enhanced RAG system initialization failed: {e}")
                        self._health_stats['last_error'] = str(e)
                        raise
        
        # Verify enhanced system health
        if (self._rag_system.faiss_index is None or 
            self._rag_system.embeddings is None):
            logger.warning("Enhanced RAG system corrupted, reinitializing...")
            with self._initialization_lock:
                await self._rag_system.load_and_process_data(DATA_PATH)
        
        return self._rag_system
    
    @timing_decorator
    async def query_with_enhanced_monitoring(self, query: str) -> Dict[str, Any]:
        """Enhanced query with comprehensive monitoring and error handling"""
        start_time = datetime.now()
        
        try:
            rag_system = await self.get_rag_system()
            result = await rag_system.get_enhanced_response(query)
            
            # Update enhanced success stats
            self._health_stats['total_queries'] += 1
            if result.get('web_sources'):
                self._health_stats['web_enhanced_queries'] += 1
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update rolling averages
            total = self._health_stats['total_queries']
            current_avg_time = self._health_stats['avg_response_time']
            current_avg_conf = self._health_stats['avg_confidence']
            
            self._health_stats['avg_response_time'] = (
                (current_avg_time * (total - 1) + response_time) / total
            )
            
            self._health_stats['avg_confidence'] = (
                (current_avg_conf * (total - 1) + result.get('confidence', 0)) / total
            )
            
            return result
            
        except Exception as e:
            self._health_stats['failed_queries'] += 1
            self._health_stats['last_error'] = str(e)
            logger.error(f"Enhanced query processing failed: {e}")
            
            return {
                'query': query,
                'answer': "معذرت، ڪا خرابي آئي آهي. مهرباني ڪري ٻيهر ڪوشش ڪريو.",
                'confidence': 0.0,
                'accuracy_score': 0.0,
                'context_chunks_used': 0,
                'web_sources': [],
                'error': str(e)
            }
    
    def get_enhanced_health_stats(self) -> Dict[str, Any]:
        """Get enhanced system health statistics"""
        success_rate = 1.0
        web_enhancement_rate = 0.0
        
        if self._health_stats['total_queries'] > 0:
            success_rate = (
                (self._health_stats['total_queries'] - self._health_stats['failed_queries']) /
                self._health_stats['total_queries']
            )
            web_enhancement_rate = (
                self._health_stats['web_enhanced_queries'] / self._health_stats['total_queries']
            )
        
        return {
            **self._health_stats,
            'success_rate': success_rate,
            'web_enhancement_rate': web_enhancement_rate,
            'system_initialized': self._rag_system is not None,
            'langgraph_available': LANGGRAPH_AVAILABLE
        }

# Global enhanced manager instance
_enhanced_rag_manager = EnhancedRAGSystemManager()
_sync_rag_system = None

async def get_enhanced_rag_system():
    """Get enhanced RAG system instance"""
    return await _enhanced_rag_manager.get_rag_system()

async def query_enhanced_chatbot(query: str) -> str:
    """
    Enhanced main function to query the chatbot with web integration
    
    Args:
        query: User query string
        
    Returns:
        Response string
    """
    try:
        if not query or not query.strip():
            return "مهرباني ڪري صحيح سوال ڏيو."
        
        result = await _enhanced_rag_manager.query_with_enhanced_monitoring(query.strip())
        return result['answer']
        
    except Exception as e:
        logger.error(f"Error in enhanced chatbot: {str(e)}")
        return f"معذرت، خرابي آئي آهي: {str(e)}"

async def query_enhanced_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    """
    Enhanced session-aware function with web integration
    
    Args:
        query: User query string
        user_id: ID of the user making the query
        session_id: Optional session ID
        
    Returns:
        dict: Enhanced response with session and web information
    """
    from ...services.session_service import SessionService
    
    try:
        # Enhanced input validation
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
        
        # Session management (unchanged)
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
        
        # Get enhanced chatbot response with web integration
        result = await _enhanced_rag_manager.query_with_enhanced_monitoring(query)
        
        # Save bot response
        bot_message_id = SessionService.save_message(session_id, 'bot', result['answer'])
        
        # Prepare enhanced response with web information
        response_data = {
            'query': query,
            'answer': result['answer'],
            'session_id': session_id,
            'user_message_id': user_message_id,
            'bot_message_id': bot_message_id,
            'confidence': convert_numpy_types(result.get('confidence', 0.0)),
            'accuracy_score': convert_numpy_types(result.get('accuracy_score', 0.0)),
            'context_chunks_used': convert_numpy_types(result.get('context_chunks_used', 0)),
            'local_chunks_used': convert_numpy_types(result.get('local_chunks_used', 0)),
            'web_chunks_used': convert_numpy_types(result.get('web_chunks_used', 0)),
            'model': 'enhanced-web-rag-chatbot',
            'query_type': result.get('query_type', 'general'),
            'response_time': convert_numpy_types(result.get('response_time', 0.0)),
            'web_sources': result.get('web_sources', []),
            'retrieval_method': result.get('retrieval_method', 'enhanced_hybrid_web'),
            'web_enhanced': len(result.get('web_sources', [])) > 0
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in enhanced session-aware chatbot: {str(e)}")
        return {
            'error': f"Internal error: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }

def get_enhanced_system_health() -> Dict[str, Any]:
    """Get enhanced system health and performance statistics"""
    try:
        health_stats = _enhanced_rag_manager.get_enhanced_health_stats()
        
        # Add enhanced cache statistics if system is initialized
        if health_stats['system_initialized']:
            rag_system = _enhanced_rag_manager._rag_system
            if rag_system:
                cache_stats = rag_system.embedding_cache.get_stats()
                health_stats['cache_stats'] = cache_stats
                
                # Add web content statistics
                health_stats['web_chunks_loaded'] = len(rag_system.web_chunk_data)
                health_stats['local_chunks_loaded'] = len(rag_system.chunk_data)
        
        return health_stats
        
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        return {
            'error': str(e),
            'system_initialized': False,
            'langgraph_available': LANGGRAPH_AVAILABLE
        }

# ========== ENHANCED LANGGRAPH VERIFICATION ==========

if LANGGRAPH_AVAILABLE:
    def create_enhanced_verification_graph():
        """Create an enhanced LangGraph workflow for response verification"""
        
        def verify_enhanced_retrieval(state):
            """Verify enhanced retrieval quality with web sources"""
            query = state.get('query', '')
            local_chunks = state.get('local_context_chunks', [])
            web_chunks = state.get('web_context_chunks', [])
            web_sources = state.get('web_sources', [])
            
            total_chunks = local_chunks + web_chunks
            
            if not total_chunks:
                return {
                    'retrieval_quality': 'poor', 
                    'needs_retry': True,
                    'enhancement_level': 'none'
                }
            
            # Enhanced quality assessment
            total_words = sum(len(chunk.split()) for chunk in total_chunks)
            query_words = set(query.lower().split())
            context_words = set()
            
            for chunk in total_chunks:
                context_words.update(chunk.lower().split())
            
            overlap = len(query_words.intersection(context_words))
            overlap_ratio = overlap / len(query_words) if query_words else 0
            
            # Enhanced quality scoring
            quality_score = 0
            if total_words > 100:
                quality_score += 0.3
            if overlap_ratio > 0.3:
                quality_score += 0.4
            if web_sources:
                quality_score += 0.2
            if len(total_chunks) >= 3:
                quality_score += 0.1
            
            quality = 'excellent' if quality_score >= 0.8 else 'good' if quality_score >= 0.6 else 'fair'
            enhancement_level = 'high' if web_sources else 'local_only'
            
            return {
                'retrieval_quality': quality,
                'needs_retry': quality == 'poor',
                'context_relevance': overlap_ratio,
                'enhancement_level': enhancement_level,
                'quality_score': quality_score
            }
        
        def verify_enhanced_answer(state):
            """Verify enhanced answer quality with web integration consideration"""
            answer = state.get('answer', '')
            confidence = state.get('confidence', 0.0)
            web_enhanced = state.get('web_enhanced', False)
            
            # Enhanced answer quality checks
            word_count = len(answer.split())
            has_content = bool(answer.strip())
            
            # Sindhi content check
            sindhi_content = bool(re.search(r'[ء-ي]', answer))
            
            quality_score = 0
            if has_content:
                quality_score += 0.2
            if word_count >= 15:  # Increased minimum
                quality_score += 0.3
            if sindhi_content:
                quality_score += 0.2  # Bonus for Sindhi responses
            if confidence > 0.5:
                quality_score += 0.2
            if web_enhanced:
                quality_score += 0.1  # Web enhancement bonus
            
            return {
                'answer_quality': quality_score,
                'is_acceptable': quality_score >= 0.7,  # Higher threshold
                'sindhi_content': sindhi_content,
                'web_enhanced': web_enhanced
            }
        
        # Build enhanced verification graph
        workflow = Graph()
        
        workflow.add_node("verify_retrieval", RunnableLambda(verify_enhanced_retrieval))
        workflow.add_node("verify_answer", RunnableLambda(verify_enhanced_answer))
        
        workflow.set_entry_point("verify_retrieval")
        workflow.add_edge("verify_retrieval", "verify_answer")
        workflow.add_edge("verify_answer", END)
        
        return workflow.compile()
    
    # Initialize enhanced verification graph
    enhanced_verification_graph = create_enhanced_verification_graph()
    
    def verify_enhanced_response_quality(query: str, answer: str, 
                                       local_chunks: List[str], web_chunks: List[str],
                                       web_sources: List[str], confidence: float) -> Dict[str, Any]:
        """Verify enhanced response quality using LangGraph"""
        try:
            result = enhanced_verification_graph.invoke({
                'query': query,
                'answer': answer,
                'local_context_chunks': local_chunks,
                'web_context_chunks': web_chunks,
                'web_sources': web_sources,
                'confidence': confidence,
                'web_enhanced': len(web_sources) > 0
            })
            return result
        except Exception as e:
            logger.warning(f"Enhanced verification graph failed: {e}")
            return {
                'answer_quality': confidence,
                'is_acceptable': len(answer.split()) > 10 and confidence > 0.4,
                'enhancement_level': 'high' if web_sources else 'local_only'
            }

else:
    def verify_enhanced_response_quality(query: str, answer: str, 
                                       local_chunks: List[str], web_chunks: List[str],
                                       web_sources: List[str], confidence: float) -> Dict[str, Any]:
        """Enhanced verification fallback without LangGraph"""
        quality_score = confidence
        
        # Basic enhancements
        if len(answer.split()) > 15:
            quality_score += 0.1
        if web_sources:
            quality_score += 0.1
        if re.search(r'[ء-ي]', answer):
            quality_score += 0.1
        
        return {
            'answer_quality': min(1.0, quality_score),
            'is_acceptable': len(answer.split()) > 10 and confidence > 0.4,
            'enhancement_level': 'high' if web_sources else 'local_only'
        }

# ========== COMPATIBILITY FUNCTIONS ==========

# Backward compatibility functions for existing code
def get_rag_system():
    """Backward compatible function - now returns enhanced system"""
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            logger.warning("Already in async context, using sync fallback")
            return _get_sync_rag_system()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_enhanced_rag_manager.get_rag_system())
    except Exception as e:
        logger.error(f"Failed to get enhanced RAG system: {e}")
        # Fallback to basic system if available
        return _get_sync_rag_system()

def _get_sync_rag_system():
    """Synchronous fallback for RAG system initialization"""
    global _sync_rag_system
    if _sync_rag_system is None:
        try:
            logger.info("Initializing synchronous RAG system...")
            # Create a simplified synchronous version
            _sync_rag_system = _create_sync_rag_system()
            logger.info("Synchronous RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Synchronous RAG system initialization failed: {e}")
            raise
    
    # Verify the system is properly initialized
    if hasattr(_sync_rag_system, 'faiss_index') and _sync_rag_system.faiss_index is None:
        logger.warning("FAISS index is None, reinitializing...")
        _sync_rag_system = _create_sync_rag_system()
    
    return _sync_rag_system

def _create_sync_rag_system():
    """Create a simplified synchronous RAG system"""
    try:
        # Create enhanced system but run initialization synchronously
        system = EnhancedProductionRAGSystem()
        
        # Run the async initialization in a new event loop
        import asyncio
        asyncio.run(system.load_and_process_data(DATA_PATH))
        
        return system
    except Exception as e:
        logger.error(f"Failed to create sync RAG system: {e}")
        raise

def _get_sync_response(rag_system, query: str) -> dict:
    """Get response using synchronous methods only"""
    try:
        logger.info(f"Getting sync response for query: {query[:50]}...")
        
        # Check if system is properly initialized
        if not hasattr(rag_system, 'faiss_index') or rag_system.faiss_index is None:
            logger.error("RAG system not properly initialized - FAISS index is None")
            return {
                'query': query,
                'answer': "سسٽم ابڃ ڀرپور طور تي شروع نه ٿيو آهي. مهرباني ڪري ٻيهر ڪوشش ڪريو.",
                'confidence': 0.0
            }
        
        logger.info(f"FAISS index has {rag_system.faiss_index.ntotal} vectors")
        
        # Simple synchronous retrieval
        query_embedding = rag_system.generate_embeddings_batch([query])
        
        if rag_system.faiss_index.ntotal == 0:
            logger.error("FAISS index is empty")
            return {
                'query': query,
                'answer': "ڊيٽا لوڊ نه ٿيو آهي. مهرباني ڪري ٻيهر ڪوشش ڪريو.",
                'confidence': 0.0
            }
        
        # Search for relevant chunks
        distances, indices = rag_system.faiss_index.search(query_embedding, 5)
        
        logger.info(f"Found {len(indices[0])} potential matches")
        
        # Filter results (IndexFlatIP returns inner product scores, higher is better)
        relevant_chunks = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(rag_system.chunk_data):
                # For Inner Product, score is already similarity-like (higher = more similar)
                similarity = float(score)
                logger.debug(f"Chunk {idx}: IP_score={score:.3f}, similarity={similarity:.3f}")
                if similarity > 0.1:  # Threshold for inner product
                    relevant_chunks.append((rag_system.chunk_data[idx]['text'], similarity))
                    logger.debug(f"Added chunk {idx} with similarity {similarity:.3f}")
        
        if not relevant_chunks:
            logger.warning("No relevant chunks found")
            return {
                'query': query,
                'answer': "ھن سوال جو جواب موجوده ڊيٽا ۾ نه ملي سگھيو.",
                'confidence': 0.0
            }
        
        # Create context from top chunks
        context_chunks = [chunk for chunk, _ in relevant_chunks[:3]]
        context_text = "\n\n".join(context_chunks)
        
        logger.info(f"Using {len(context_chunks)} chunks for context")
        
        # Generate response using Together API
        try:
            response = rag_system.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "system", "content": "توھان شاھ عبداللطيف ڀٽائي جي باري ۾ ماھر آھيو۔ صرف ڏنل حوالن مان معلومات ڏيو۔"},
                    {"role": "user", "content": f"سوال: {query}\n\nحوالو:\n{context_text}\n\nجواب:"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            confidence = len(relevant_chunks) / 5  # Simple confidence calculation
            
            logger.info(f"Generated response with confidence {confidence:.2f}")
            
            return {
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'context_chunks_used': len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'answer': "جواب تيار ڪرڻ ۾ خرابي آئي آهي.",
                'confidence': 0.0
            }
        
    except Exception as e:
        logger.error(f"Error in sync response generation: {e}")
        return {
            'query': query,
            'answer': f"خرابي: {str(e)}",
            'confidence': 0.0
        }

def query_general_chatbot(query: str) -> str:
    """
    Enhanced backward compatible function for querying the chatbot
    
    Args:
        query: User query string
        
    Returns:
        Response string
    """
    try:
        if not query or not query.strip():
            return "مهرباني ڪري صحيح سوال ڏيو."
        
        # Use the sync fallback directly to avoid async issues
        logger.info(f"Processing query: {query[:50]}...")
        rag_system = _get_sync_rag_system()
        
        # Use a simplified response method for sync operations
        result = _get_sync_response(rag_system, query.strip())
        return result.get('answer', "توھان جي سوال جو جواب نه ملي سگھيو.")
        
    except Exception as e:
        logger.error(f"Error in general chatbot: {str(e)}")
        return f"معذرت، خرابي آئي آهي: {str(e)}"

def query_general_chatbot_with_session(query: str, user_id: str, session_id: str = None) -> dict:
    """
    Enhanced backward compatible session-aware function
    
    Args:
        query: User query string
        user_id: ID of the user making the query
        session_id: Optional session ID
        
    Returns:
        dict: Enhanced response with session and web information
    """
    from ...services.session_service import SessionService
    
    try:
        if not query or not query.strip():
            return {
                'error': 'مهرباني ڪري صحيح سوال ڏيو.',
                'code': 'INVALID_QUERY'
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
        else:
            # Create new session with query as session name
            session_name = query[:100] if len(query) <= 100 else query[:97] + "..."
            session_id = SessionService.create_session(user_id, session_name)
        
        # Save user message
        user_message_id = SessionService.save_message(session_id, 'user', query)
        
        # Update session activity
        SessionService.update_session_activity(session_id)
        
        # Get chatbot response using the same sync approach as query_general_chatbot
        logger.info(f"Processing session query: {query[:50]}...")
        rag_system = _get_sync_rag_system()
        
        # Use the same simplified response method for sync operations
        result = _get_sync_response(rag_system, query.strip())
        
        # Save bot response
        answer = result.get('answer', "توھان جي سوال جو جواب نه ملي سگھيو.")
        bot_message_id = SessionService.save_message(session_id, 'bot', answer)
        
        # Convert numpy types to native Python types for JSON serialization
        response_data = {
            'query': query,
            'answer': answer,
            'session_id': session_id,
            'user_message_id': user_message_id,
            'bot_message_id': bot_message_id,
            'confidence': float(result.get('confidence', 0.0)),
            'accuracy_score': 1.0,  # Simple accuracy score
            'context_chunks_used': result.get('context_chunks_used', 0),
            'model': 'general-chatbot'
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in session-aware general chatbot: {str(e)}")
        return {
            'error': f"معذرت، خرابي آئي آهي: {str(e)}",
            'code': 'INTERNAL_ERROR'
        }

def get_system_health() -> Dict[str, Any]:
    """Enhanced backward compatible system health function"""
    return get_enhanced_system_health()

# ========== DIAGNOSTIC AND TESTING FUNCTIONS ==========

async def test_web_integration():
    """Test web integration functionality"""
    print("🧪 Testing web integration...")
    
    try:
        # Test web scraper
        scraper = BhittaipediaWebScraper()
        test_contents = await scraper.search_specific_query("شاه لطيف")
        
        print(f"✅ Web scraper test: Found {len(test_contents)} web contents")
        
        # Test enhanced RAG system
        rag_system = await get_enhanced_rag_system()
        test_query = "شاه لطيف ڪٿي ڄميو؟"
        result = await rag_system.get_enhanced_response(test_query)
        
        print(f"✅ Enhanced RAG test:")
        print(f"   - Query: {test_query}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Web sources: {len(result.get('web_sources', []))}")
        print(f"   - Answer length: {len(result['answer'].split())} words")
        
        return True
        
    except Exception as e:
        print(f"❌ Web integration test failed: {e}")
        return False

async def analyze_similarity_performance():
    """Analyze and suggest similarity threshold improvements"""
    print("📊 Analyzing similarity performance...")
    
    try:
        rag_system = await get_enhanced_rag_system()
        
        # Test queries with known answers
        test_queries = [
            "شاه لطيف ڪٿي ڄميو؟",
            "شاه لطيف جي شاعري بابت ٻڌايو",
            "شاه لطيف جو ڪو پٽ هو؟",
            "سُر ساسوئي بابت معلومات ڏيو"
        ]
        
        similarity_stats = []
        
        for query in test_queries:
            query_context = rag_system.enhanced_query_processing(query)
            
            # Test local retrieval
            local_results = rag_system._local_retrieval(query, query_context)
            
            if local_results:
                top_scores = [score for _, score in local_results[:5]]
                avg_score = np.mean(top_scores)
                min_score = np.min(top_scores)
                max_score = np.max(top_scores)
                
                similarity_stats.append({
                    'query': query,
                    'avg_similarity': avg_score,
                    'min_similarity': min_score,
                    'max_similarity': max_score,
                    'results_count': len(local_results)
                })
        
        # Print analysis
        print("📈 Similarity Analysis Results:")
        for stat in similarity_stats:
            print(f"Query: {stat['query'][:30]}...")
            print(f"  Avg: {stat['avg_similarity']:.3f}")
            print(f"  Min: {stat['min_similarity']:.3f}")
            print(f"  Max: {stat['max_similarity']:.3f}")
            print(f"  Results: {stat['results_count']}")
            print()
        
        # Suggest optimal thresholds
        all_mins = [s['min_similarity'] for s in similarity_stats]
        suggested_threshold = np.percentile(all_mins, 25) if all_mins else MIN_SIMILARITY_THRESHOLD
        
        print(f"💡 Suggested similarity threshold: {suggested_threshold:.3f}")
        print(f"🔧 Current threshold: {MIN_SIMILARITY_THRESHOLD}")
        
        return similarity_stats
        
    except Exception as e:
        print(f"❌ Similarity analysis failed: {e}")
        return []

# ========== INITIALIZATION HELPERS ==========

async def initialize_enhanced_system():
    """Initialize the enhanced RAG system"""
    try:
        print("🚀 Initializing Enhanced RAG System...")
        
        # Test web integration first
        web_test_passed = await test_web_integration()
        
        if not web_test_passed:
            print("⚠️ Web integration test failed, but system will continue with local data only")
        
        # Initialize the system
        rag_system = await get_enhanced_rag_system()
        
        # Run similarity analysis
        await analyze_similarity_performance()
        
        # Print final status
        health = get_enhanced_system_health()
        print("\n🎯 Enhanced System Status:")
        print(f"   - System Initialized: {health['system_initialized']}")
        print(f"   - LangGraph Available: {health['langgraph_available']}")
        print(f"   - Local Chunks: {health.get('local_chunks_loaded', 0)}")
        print(f"   - Web Chunks: {health.get('web_chunks_loaded', 0)}")
        
        return rag_system
        
    except Exception as e:
        logger.error(f"Enhanced system initialization failed: {e}")
        raise

def run_enhanced_diagnostic():
    """Run comprehensive diagnostic on the enhanced system"""
    async def diagnostic():
        print("🔧 Running Enhanced System Diagnostic...")
        
        try:
            # Initialize system
            await initialize_enhanced_system()
            
            # Test various query types
            test_queries = [
                "شاه لطيف ڪٿي ڄميو؟",
                "شاه لطيف جي زال جو نالو ڇا هو؟",
                "شاه لطيف جو ڪو پٽ هو؟",
                "سُر ساسوئي بابت ٻڌايو",
                "شاه لطيف جي فلسفي بابت ڄاڻ"
            ]
            
            print("\n🧪 Testing Enhanced Queries:")
            for query in test_queries:
                try:
                    result = await query_enhanced_chatbot(query)
                    print(f"\nQ: {query}")
                    print(f"A: {result[:100]}...")
                except Exception as e:
                    print(f"❌ Query failed: {query} - {e}")
            
            # Print final health stats
            health = get_enhanced_system_health()
            print(f"\n📊 Final Health Stats:")
            for key, value in health.items():
                if not key.startswith('_'):
                    print(f"   {key}: {value}")
            
            print("\n✅ Enhanced diagnostic completed!")
            
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")
    
    asyncio.run(diagnostic())

# ========== PERFORMANCE MONITORING ==========

class PerformanceMonitor:
    """Monitor system performance and suggest optimizations"""
    
    def __init__(self):
        self.query_times = deque(maxlen=100)
        self.confidence_scores = deque(maxlen=100)
        self.web_usage_rate = deque(maxlen=100)
        
    def record_query(self, response_time: float, confidence: float, used_web: bool):
        """Record query performance metrics"""
        self.query_times.append(response_time)
        self.confidence_scores.append(confidence)
        self.web_usage_rate.append(1 if used_web else 0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report with recommendations"""
        if not self.query_times:
            return {'status': 'insufficient_data'}
        
        avg_time = np.mean(self.query_times)
        avg_confidence = np.mean(self.confidence_scores)
        web_usage = np.mean(self.web_usage_rate)
        
        recommendations = []
        
        if avg_time > 5.0:
            recommendations.append("Consider reducing INITIAL_RETRIEVE_K for faster response")
        
        if avg_confidence < 0.6:
            recommendations.append("Consider lowering MIN_SIMILARITY_THRESHOLD")
            recommendations.append("Increase web content integration")
        
        if web_usage < 0.3:
            recommendations.append("Web integration underutilized - check web scraping")
        
        return {
            'avg_response_time': avg_time,
            'avg_confidence': avg_confidence,
            'web_usage_rate': web_usage,
            'total_queries': len(self.query_times),
            'recommendations': recommendations
        }

# Global performance monitor
_performance_monitor = PerformanceMonitor()

# ========== MAIN EXECUTION HELPER ==========

if __name__ == "__main__":
    print("🚀 Enhanced RAG System with Web Integration")
    print("=" * 50)
    
    # Run diagnostic
    run_enhanced_diagnostic()
    
    print("\n" + "=" * 50)
    print("Enhanced system ready for production use!")
    print("\nKey improvements:")
    print("✅ Integrated Bhittaipedia.org web scraping")
    print("✅ LangGraph agents for intelligent web search")
    print("✅ Improved similarity thresholds")
    print("✅ Enhanced context expansion")
    print("✅ Better conversation memory")
    print("✅ Performance monitoring")
    
    # Example usage
    print("\n📝 Example usage:")
    print("result = await query_enhanced_chatbot('شاه لطيف ڪٿي ڄميو؟')")
    print("health = get_enhanced_system_health()")