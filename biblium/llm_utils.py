# -*- coding: utf-8 -*-
"""
Biblium LLM Utilities Module

Optimized LLM functions with:
- Multi-provider support (HuggingFace, OpenAI, Anthropic)
- Response caching (disk and memory)
- Batch processing
- Async support
- Rate limiting
- Retry with exponential backoff
- Better prompt templates for bibliometric tasks

@author: Claude (Anthropic) for Lan.Umek
@version: 2.3.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    
    # Provider settings
    provider: str = "huggingface"  # "huggingface", "openai", "anthropic"
    model: Optional[str] = None
    api_key: Optional[str] = None
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    
    # Caching settings
    cache_enabled: bool = True
    cache_dir: Optional[str] = None  # None = memory only
    cache_ttl: int = 86400 * 7  # 7 days
    
    # Rate limiting
    rate_limit_rpm: int = 60  # requests per minute
    rate_limit_tpm: int = 100000  # tokens per minute
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Batch settings
    batch_size: int = 5
    batch_delay: float = 0.5
    
    # Fallback models per provider
    fallback_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "huggingface": [
            "google/gemma-2-2b-it",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
        ],
    })


# Global default config
_default_config = LLMConfig()


def get_default_config() -> LLMConfig:
    """Get the default LLM configuration."""
    return _default_config


def set_default_config(config: LLMConfig):
    """Set the default LLM configuration."""
    global _default_config
    _default_config = config


# =============================================================================
# CACHING
# =============================================================================

class LLMCache:
    """
    LLM response cache with memory and disk persistence.
    
    Features:
    - Memory cache with LRU eviction
    - Optional disk persistence
    - TTL-based expiration
    - Hash-based keys for prompt deduplication
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 1000,
        ttl: int = 86400 * 7,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.ttl = ttl
        self._memory_cache: Dict[str, Tuple[str, float]] = {}
        self._access_order: List[str] = []
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_key(self, prompt: str, model: str, provider: str, **kwargs) -> str:
        """Create a hash key from prompt and settings."""
        key_data = {
            "prompt": prompt,
            "model": model,
            "provider": provider,
            **{k: v for k, v in sorted(kwargs.items()) if v is not None}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(self, prompt: str, model: str, provider: str, **kwargs) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._hash_key(prompt, model, provider, **kwargs)
        
        # Check memory cache
        if key in self._memory_cache:
            response, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl:
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return response
            else:
                # Expired
                del self._memory_cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        
        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if time.time() - data.get("timestamp", 0) < self.ttl:
                        response = data["response"]
                        # Add to memory cache
                        self._set_memory(key, response)
                        return response
                    else:
                        # Expired, remove file
                        cache_file.unlink(missing_ok=True)
                except Exception:
                    pass
        
        return None
    
    def _set_memory(self, key: str, response: str):
        """Set item in memory cache with LRU eviction."""
        # Evict oldest if at capacity
        while len(self._memory_cache) >= self.max_memory_items and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._memory_cache.pop(oldest_key, None)
        
        self._memory_cache[key] = (response, time.time())
        self._access_order.append(key)
    
    def set(self, prompt: str, model: str, provider: str, response: str, **kwargs):
        """Cache a response."""
        key = self._hash_key(prompt, model, provider, **kwargs)
        
        # Set in memory
        self._set_memory(key, response)
        
        # Set on disk
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                data = {
                    "response": response,
                    "timestamp": time.time(),
                    "model": model,
                    "provider": provider,
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception:
                pass
    
    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()
        self._access_order.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.json"):
                f.unlink(missing_ok=True)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_count = 0
        if self.cache_dir:
            disk_count = len(list(self.cache_dir.glob("*.json")))
        return {
            "memory_items": len(self._memory_cache),
            "disk_items": disk_count,
            "max_memory_items": self.max_memory_items,
            "ttl_seconds": self.ttl,
        }


# Global cache instance
_global_cache: Optional[LLMCache] = None


def get_cache(config: Optional[LLMConfig] = None) -> LLMCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        cfg = config or _default_config
        _global_cache = LLMCache(
            cache_dir=cfg.cache_dir,
            ttl=cfg.cache_ttl,
        )
    return _global_cache


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Supports both requests-per-minute and tokens-per-minute limits.
    """
    
    def __init__(self, rpm: int = 60, tpm: int = 100000):
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: List[float] = []
        self._token_counts: List[Tuple[float, int]] = []
    
    def _clean_old_entries(self, window: float = 60.0):
        """Remove entries older than the window."""
        cutoff = time.time() - window
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]
    
    def wait_if_needed(self, estimated_tokens: int = 100):
        """Wait if rate limit would be exceeded."""
        self._clean_old_entries()
        
        # Check RPM
        if len(self._request_times) >= self.rpm:
            wait_time = self._request_times[0] + 60 - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
                self._clean_old_entries()
        
        # Check TPM
        current_tokens = sum(c for _, c in self._token_counts)
        if current_tokens + estimated_tokens > self.tpm:
            # Wait until oldest tokens expire
            if self._token_counts:
                wait_time = self._token_counts[0][0] + 60 - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)
                    self._clean_old_entries()
    
    def record_request(self, tokens_used: int = 100):
        """Record a completed request."""
        now = time.time()
        self._request_times.append(now)
        self._token_counts.append((now, tokens_used))


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[LLMConfig] = None) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        cfg = config or _default_config
        _rate_limiter = RateLimiter(rpm=cfg.rate_limit_rpm, tpm=cfg.rate_limit_tpm)
    return _rate_limiter


# =============================================================================
# RETRY LOGIC
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
):
    """
    Decorator for retry with exponential backoff.
    
    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts.
    initial_delay : float
        Initial delay between retries in seconds.
    backoff_factor : float
        Multiplier for delay after each retry.
    exceptions : tuple
        Exception types to catch and retry.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# PROVIDER CLIENTS
# =============================================================================

class BaseProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider."""
    
    DEFAULT_MODELS = [
        "google/gemma-2-2b-it",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    def _get_client(self, model: str):
        try:
            from huggingface_hub import InferenceClient
            return InferenceClient(model=model, token=self.api_key)
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        model = model or self.model or self.DEFAULT_MODELS[0]
        client = self._get_client(model)
        
        try:
            # Try chat completion first
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return self._extract_response(response)
        except Exception as e:
            # Fallback to text generation
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.strip() if isinstance(response, str) else str(response).strip()
            except Exception:
                raise e
    
    def _extract_response(self, response) -> str:
        """Extract text from chat completion response."""
        try:
            return response.choices[0].message.content.strip()
        except (AttributeError, IndexError):
            try:
                return response.choices[0].message["content"].strip()
            except:
                return str(response).strip()
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generation using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    DEFAULT_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
        return self._client
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        model = model or self.model or self.DEFAULT_MODELS[0]
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generation."""
        model = kwargs.get("model") or self.model or self.DEFAULT_MODELS[0]
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            # Fallback to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""
    
    DEFAULT_MODELS = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic is required. Install with: pip install anthropic")
        return self._client
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        model = model or self.model or self.DEFAULT_MODELS[0]
        client = self._get_client()
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text.strip()
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generation."""
        model = kwargs.get("model") or self.model or self.DEFAULT_MODELS[0]
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 512),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.content[0].text.strip()
        except ImportError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))


def get_provider(
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseProvider:
    """Get a provider instance."""
    providers = {
        "huggingface": HuggingFaceProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    return providers[provider](api_key=api_key, model=model)


# =============================================================================
# MAIN LLM FUNCTIONS
# =============================================================================

def invoke_llm(
    prompt: str,
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    use_cache: bool = True,
    use_rate_limit: bool = True,
    **gen_kwargs,
) -> str:
    """
    Invoke an LLM with caching, rate limiting, and retry logic.
    
    Parameters
    ----------
    prompt : str
        The input prompt.
    model : str, optional
        Model identifier. Uses config default if not specified.
    provider : str, default "huggingface"
        Provider to use: "huggingface", "openai", or "anthropic".
    api_key : str, optional
        API key. Can also be set via environment variables.
    config : LLMConfig, optional
        Configuration object. Uses global default if not specified.
    use_cache : bool, default True
        Whether to use response caching.
    use_rate_limit : bool, default True
        Whether to apply rate limiting.
    **gen_kwargs :
        Additional generation parameters (max_tokens, temperature, etc.)
    
    Returns
    -------
    str
        Generated text response.
    
    Examples
    --------
    >>> response = invoke_llm(
    ...     "Summarize this text: ...",
    ...     provider="openai",
    ...     api_key="sk-...",
    ...     max_tokens=256,
    ... )
    """
    cfg = config or _default_config
    
    # Resolve API key from environment if not provided
    if api_key is None:
        api_key = _get_api_key_from_env(provider)
    
    # Resolve model
    model = model or cfg.model or cfg.fallback_models.get(provider, [""])[0]
    
    # Merge generation kwargs with config defaults
    gen_kwargs.setdefault("max_tokens", cfg.max_tokens)
    gen_kwargs.setdefault("temperature", cfg.temperature)
    
    # Check cache
    cache = get_cache(cfg) if use_cache and cfg.cache_enabled else None
    if cache:
        cached = cache.get(prompt, model, provider, **gen_kwargs)
        if cached is not None:
            return cached
    
    # Rate limiting
    if use_rate_limit:
        limiter = get_rate_limiter(cfg)
        estimated_tokens = len(prompt.split()) + gen_kwargs.get("max_tokens", 512)
        limiter.wait_if_needed(estimated_tokens)
    
    # Get provider and generate
    prov = get_provider(provider, api_key=api_key, model=model)
    
    @retry_with_backoff(
        max_retries=cfg.max_retries,
        initial_delay=cfg.retry_delay,
        backoff_factor=cfg.retry_backoff,
    )
    def _generate():
        return prov.generate(prompt, model=model, **gen_kwargs)
    
    response = _generate()
    
    # Record request for rate limiting
    if use_rate_limit:
        tokens_used = len(prompt.split()) + len(response.split())
        get_rate_limiter(cfg).record_request(tokens_used)
    
    # Cache response
    if cache:
        cache.set(prompt, model, provider, response, **gen_kwargs)
    
    return response


async def invoke_llm_async(
    prompt: str,
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    use_cache: bool = True,
    **gen_kwargs,
) -> str:
    """
    Async version of invoke_llm.
    
    Same parameters as invoke_llm but runs asynchronously.
    """
    cfg = config or _default_config
    
    if api_key is None:
        api_key = _get_api_key_from_env(provider)
    
    model = model or cfg.model or cfg.fallback_models.get(provider, [""])[0]
    gen_kwargs.setdefault("max_tokens", cfg.max_tokens)
    gen_kwargs.setdefault("temperature", cfg.temperature)
    
    # Check cache
    cache = get_cache(cfg) if use_cache and cfg.cache_enabled else None
    if cache:
        cached = cache.get(prompt, model, provider, **gen_kwargs)
        if cached is not None:
            return cached
    
    prov = get_provider(provider, api_key=api_key, model=model)
    response = await prov.generate_async(prompt, model=model, **gen_kwargs)
    
    if cache:
        cache.set(prompt, model, provider, response, **gen_kwargs)
    
    return response


def _get_api_key_from_env(provider: str) -> Optional[str]:
    """Get API key from environment variable."""
    env_vars = {
        "huggingface": ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }
    for var in env_vars.get(provider, []):
        key = os.environ.get(var)
        if key:
            return key
    return None


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def invoke_llm_batch(
    prompts: List[str],
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    max_workers: int = 4,
    show_progress: bool = True,
    **gen_kwargs,
) -> List[str]:
    """
    Process multiple prompts in parallel with threading.
    
    Parameters
    ----------
    prompts : list of str
        List of prompts to process.
    max_workers : int, default 4
        Maximum number of parallel workers.
    show_progress : bool, default True
        Show progress indicator.
    **kwargs :
        Same as invoke_llm.
    
    Returns
    -------
    list of str
        Responses in same order as prompts.
    """
    results = [None] * len(prompts)
    
    def process_one(idx: int, prompt: str) -> Tuple[int, str]:
        response = invoke_llm(
            prompt,
            model=model,
            provider=provider,
            api_key=api_key,
            config=config,
            **gen_kwargs,
        )
        return idx, response
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, i, p): i
            for i, p in enumerate(prompts)
        }
        
        completed = 0
        for future in as_completed(futures):
            idx, response = future.result()
            results[idx] = response
            completed += 1
            if show_progress:
                print(f"\r  Processing: {completed}/{len(prompts)}", end="", flush=True)
        
        if show_progress:
            print()
    
    return results


async def invoke_llm_batch_async(
    prompts: List[str],
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    max_concurrent: int = 10,
    **gen_kwargs,
) -> List[str]:
    """
    Process multiple prompts with async concurrency.
    
    Parameters
    ----------
    prompts : list of str
        List of prompts to process.
    max_concurrent : int, default 10
        Maximum concurrent requests.
    **kwargs :
        Same as invoke_llm_async.
    
    Returns
    -------
    list of str
        Responses in same order as prompts.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(prompt: str) -> str:
        async with semaphore:
            return await invoke_llm_async(
                prompt,
                model=model,
                provider=provider,
                api_key=api_key,
                config=config,
                **gen_kwargs,
            )
    
    tasks = [process_one(p) for p in prompts]
    return await asyncio.gather(*tasks)


# =============================================================================
# BIBLIOMETRIC-SPECIFIC LLM FUNCTIONS
# =============================================================================

# Prompt templates optimized for bibliometric tasks
PROMPT_TEMPLATES = {
    "summarize_abstracts": """You are a research synthesis expert. Analyze the following {n_abstracts} academic abstracts and provide a coherent synthesis.

ABSTRACTS:
{abstracts}

Provide a synthesis that:
1. Identifies the main research themes and findings
2. Notes methodological approaches used
3. Highlights key conclusions and their implications
4. Is written in academic style (3-5 sentences)

SYNTHESIS:""",

    "describe_table": """You are a bibliometric analyst. Analyze the following data table and provide insights.

TABLE:
{table}

Provide a brief analysis that:
1. Summarizes the main patterns in the data
2. Highlights notable values or outliers
3. Suggests potential implications
4. Uses specific numbers from the table

Keep your response to 2-3 sentences. Be precise and data-driven.

ANALYSIS:""",

    "extract_keywords": """Extract the most important keywords/concepts from the following academic text.

TEXT:
{text}

Return ONLY a comma-separated list of 5-10 key concepts, ordered by importance.
Do not include any other text or explanation.

KEYWORDS:""",

    "classify_methodology": """Classify the research methodology described in this abstract.

ABSTRACT:
{abstract}

Classify as one of: quantitative, qualitative, mixed-methods, theoretical, review, meta-analysis, case-study, experimental, observational

Return ONLY the classification label, nothing else.

CLASSIFICATION:""",

    "identify_research_gaps": """Analyze these abstracts to identify potential research gaps.

ABSTRACTS:
{abstracts}

List 3-5 specific research gaps or future research directions suggested by these papers.
Format as a numbered list.

RESEARCH GAPS:""",

    "compare_papers": """Compare and contrast these two academic abstracts.

ABSTRACT 1:
{abstract1}

ABSTRACT 2:
{abstract2}

Provide a brief comparison covering:
1. Main similarities
2. Key differences
3. How they might complement each other

Keep response to 3-4 sentences.

COMPARISON:""",
}


def llm_summarize_abstracts(
    abstracts: List[str],
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    prompt_template: Optional[str] = None,
    **gen_kwargs,
) -> str:
    """
    Summarize multiple abstracts into a coherent synthesis.
    
    Parameters
    ----------
    abstracts : list of str
        Abstract texts to summarize.
    prompt_template : str, optional
        Custom prompt template with {abstracts} and {n_abstracts} placeholders.
    **kwargs :
        Passed to invoke_llm.
    
    Returns
    -------
    str
        Synthesized summary.
    """
    template = prompt_template or PROMPT_TEMPLATES["summarize_abstracts"]
    joined = "\n\n---\n\n".join(f"[{i+1}] {a}" for i, a in enumerate(abstracts) if a)
    prompt = template.format(abstracts=joined, n_abstracts=len(abstracts))
    
    gen_kwargs.setdefault("max_tokens", 512)
    return invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )


def llm_describe_table(
    table: Union[pd.DataFrame, str],
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    prompt_template: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    max_rows: int = 50,
    **gen_kwargs,
) -> str:
    """
    Generate a description of a data table.
    
    Parameters
    ----------
    table : DataFrame or str
        Table to describe.
    prompt_template : str, optional
        Prompt template with {table} placeholder.
    custom_prompt : str, optional
        Custom prompt. Use {data} placeholder for table data.
    max_rows : int, default 50
        Maximum rows to include in prompt.
    **kwargs :
        Passed to invoke_llm.
    
    Returns
    -------
    str
        Table description.
    """
    if isinstance(table, pd.DataFrame):
        if len(table) > max_rows:
            table = table.head(max_rows)
        table_str = table.to_markdown(index=False)
    else:
        table_str = str(table)
    
    # Use custom prompt if provided
    if custom_prompt and custom_prompt.strip():
        if "{data}" in custom_prompt:
            prompt = custom_prompt.format(data=table_str)
        elif "{table}" in custom_prompt:
            prompt = custom_prompt.format(table=table_str)
        else:
            prompt = f"{custom_prompt}\n\n{table_str}"
    else:
        template = prompt_template or PROMPT_TEMPLATES["describe_table"]
        prompt = template.format(table=table_str)
    
    gen_kwargs.setdefault("max_tokens", 256)
    return invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )


def llm_describe_plot(
    plot_type: str = "chart",
    title: str = "",
    data_summary: str = "",
    x_axis: str = "",
    y_axis: str = "",
    context: str = "",
    model: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    custom_prompt: Optional[str] = None,
    **gen_kwargs,
) -> str:
    """
    Generate a description of a plot/chart based on its metadata.
    
    Parameters
    ----------
    plot_type : str
        Type of plot (e.g., "bar chart", "line chart", "scatter plot").
    title : str
        Title of the plot.
    data_summary : str
        Summary of the data (ranges, counts, etc.).
    x_axis : str
        X-axis label.
    y_axis : str
        Y-axis label.
    context : str
        Additional context (legend, annotations, etc.).
    model : str, optional
        Model to use.
    provider : str, default "openai"
        LLM provider.
    api_key : str, optional
        API key.
    config : LLMConfig, optional
        Configuration object.
    custom_prompt : str, optional
        Custom prompt template. Use {data} placeholder for plot info.
    **gen_kwargs :
        Passed to invoke_llm.
    
    Returns
    -------
    str
        Plot description.
    """
    # Build plot data string
    plot_data = f"""Plot Type: {plot_type}
Title: {title}
X-Axis: {x_axis}
Y-Axis: {y_axis}
Data Summary: {data_summary}
Additional Context: {context}"""
    
    # Use custom prompt if provided
    if custom_prompt and custom_prompt.strip():
        if "{data}" in custom_prompt:
            prompt = custom_prompt.format(data=plot_data)
        else:
            prompt = f"{custom_prompt}\n\n{plot_data}"
    else:
        prompt = f"""Analyze this bibliometric visualization and provide insights:

{plot_data}

Provide a concise interpretation of this chart that:
1. Describes the main trend or pattern shown
2. Highlights any notable features (peaks, growth rates, anomalies)
3. Suggests what this means for the research field

Keep the response to 3-4 sentences, focused on actionable insights."""

    gen_kwargs.setdefault("max_tokens", 256)
    return invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )


def llm_extract_keywords(
    text: str,
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **gen_kwargs,
) -> List[str]:
    """
    Extract keywords from academic text.
    
    Returns
    -------
    list of str
        Extracted keywords.
    """
    prompt = PROMPT_TEMPLATES["extract_keywords"].format(text=text[:4000])
    gen_kwargs.setdefault("max_tokens", 100)
    gen_kwargs.setdefault("temperature", 0.1)
    
    response = invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )
    
    # Parse comma-separated keywords
    keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
    return keywords


def llm_classify_methodology(
    abstract: str,
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **gen_kwargs,
) -> str:
    """
    Classify the methodology of a research paper.
    
    Returns
    -------
    str
        Methodology classification.
    """
    prompt = PROMPT_TEMPLATES["classify_methodology"].format(abstract=abstract[:2000])
    gen_kwargs.setdefault("max_tokens", 20)
    gen_kwargs.setdefault("temperature", 0.0)
    
    response = invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )
    
    return response.strip().lower()


def llm_identify_research_gaps(
    abstracts: List[str],
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **gen_kwargs,
) -> str:
    """
    Identify research gaps from a set of abstracts.
    
    Returns
    -------
    str
        Identified research gaps.
    """
    joined = "\n\n".join(f"[{i+1}] {a[:500]}" for i, a in enumerate(abstracts[:10]) if a)
    prompt = PROMPT_TEMPLATES["identify_research_gaps"].format(abstracts=joined)
    gen_kwargs.setdefault("max_tokens", 400)
    
    return invoke_llm(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        config=config,
        **gen_kwargs,
    )


def llm_batch_classify(
    abstracts: List[str],
    task: Literal["methodology", "keywords", "summary"] = "methodology",
    model: Optional[str] = None,
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    max_workers: int = 4,
    show_progress: bool = True,
    **gen_kwargs,
) -> List[str]:
    """
    Batch classify/process multiple abstracts.
    
    Parameters
    ----------
    abstracts : list of str
        Abstracts to process.
    task : str
        Task type: "methodology", "keywords", or "summary".
    max_workers : int
        Parallel workers.
    
    Returns
    -------
    list of str
        Results for each abstract.
    """
    task_funcs = {
        "methodology": lambda a: llm_classify_methodology(
            a, model=model, provider=provider, api_key=api_key, config=config, **gen_kwargs
        ),
        "keywords": lambda a: ", ".join(llm_extract_keywords(
            a, model=model, provider=provider, api_key=api_key, config=config, **gen_kwargs
        )),
        "summary": lambda a: llm_summarize_abstracts(
            [a], model=model, provider=provider, api_key=api_key, config=config, **gen_kwargs
        ),
    }
    
    func = task_funcs.get(task)
    if not func:
        raise ValueError(f"Unknown task: {task}. Available: {list(task_funcs.keys())}")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, a): i for i, a in enumerate(abstracts)}
        results = [None] * len(abstracts)
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"ERROR: {str(e)}"
            completed += 1
            if show_progress:
                print(f"\r  Processing: {completed}/{len(abstracts)}", end="", flush=True)
        
        if show_progress:
            print()
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_summarize(
    texts: Union[str, List[str]],
    api_key: Optional[str] = None,
    provider: str = "huggingface",
) -> str:
    """
    Quick one-liner to summarize text(s).
    
    Examples
    --------
    >>> summary = quick_summarize(df['Abstract'].tolist()[:5], api_key="hf_...")
    """
    if isinstance(texts, str):
        texts = [texts]
    return llm_summarize_abstracts(texts, api_key=api_key, provider=provider)


def quick_describe(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    provider: str = "huggingface",
) -> str:
    """
    Quick one-liner to describe a DataFrame.
    
    Examples
    --------
    >>> description = quick_describe(my_df, api_key="hf_...")
    """
    return llm_describe_table(df, api_key=api_key, provider=provider)


# =============================================================================
# INTEGRATION WITH BIBLIUM
# =============================================================================

def add_llm_methods_to_class(cls):
    """
    Decorator to add LLM methods to BiblioAnalysis or BiblioGroupAnalysis.
    
    Usage
    -----
    @add_llm_methods_to_class
    class BiblioAnalysis:
        ...
    """
    def llm_summarize_field(
        self,
        field: str = "Abstract",
        sample_size: int = 10,
        api_key: Optional[str] = None,
        provider: str = "huggingface",
        **kwargs,
    ) -> str:
        """Summarize a text field from the dataset."""
        if field not in self.df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame")
        texts = self.df[field].dropna().sample(min(sample_size, len(self.df))).tolist()
        return llm_summarize_abstracts(texts, api_key=api_key, provider=provider, **kwargs)
    
    def llm_classify_all(
        self,
        task: str = "methodology",
        field: str = "Abstract",
        api_key: Optional[str] = None,
        provider: str = "huggingface",
        **kwargs,
    ) -> pd.Series:
        """Classify all documents using LLM."""
        abstracts = self.df[field].fillna("").tolist()
        results = llm_batch_classify(
            abstracts, task=task, api_key=api_key, provider=provider, **kwargs
        )
        return pd.Series(results, index=self.df.index, name=f"llm_{task}")
    
    cls.llm_summarize_field = llm_summarize_field
    cls.llm_classify_all = llm_classify_all
    return cls


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    # Config
    "LLMConfig",
    "get_default_config",
    "set_default_config",
    # Cache
    "LLMCache",
    "get_cache",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    # Providers
    "get_provider",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    # Main functions
    "invoke_llm",
    "invoke_llm_async",
    "invoke_llm_batch",
    "invoke_llm_batch_async",
    # Bibliometric functions
    "llm_summarize_abstracts",
    "llm_describe_table",
    "llm_extract_keywords",
    "llm_classify_methodology",
    "llm_identify_research_gaps",
    "llm_batch_classify",
    # Convenience
    "quick_summarize",
    "quick_describe",
    # Integration
    "add_llm_methods_to_class",
    # Templates
    "PROMPT_TEMPLATES",
]
