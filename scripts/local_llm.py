#!/usr/bin/env python3
"""
Interface to local LLM (Ollama).
"""

import requests
import json
import time
import logging
import re
from typing import Dict, Tuple
import tiktoken

logger = logging.getLogger(__name__)

class ValidationFailure(ValueError):
    """Custom exception to carry the failed text back to the handler."""
    def __init__(self, message, text):
        super().__init__(message)
        self.text = text

class OllamaRewriter:
    """
    Wrapper around Ollama REST API for prose rewriting.
    Features:
    - Native Tokenizer with Caching
    - Exponential Backoff Retry
    - Response Validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get("model", {}).get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get("model", {}).get('model_name', 'mistral:latest')
        self.temperature = config.get("model", {}).get('temperature', 0.65)
        self.top_p = config.get("model", {}).get('top_p', 0.9)
        self.top_k = config.get("model", {}).get('top_k', 40)
        self.num_predict = config.get("model", {}).get('num_predict', 2048)
        self.num_thread = config.get("model", {}).get('num_thread', None)
        self.seed = config.get("model", {}).get('seed', 42)
        self.timeout = config.get("model", {}).get('timeout_seconds', 300)
        self.temp_guard = None
        
        # Init token cache
        self._token_cache = {}

        # Init tokenizer
        try: 
            self.tokenizer = tiktoken.get_encoding(self.config.get("model", {}).get('tiktoken_encoding', '"cl100k_base"'))
        except Exception as e:
            logger.error(f"[LLM] Tiktoken failed to load: {e}")
            self.tokenizer = None
        
        # Pre-flight connection check
        self._verify_connection()

    def set_temp_guard(self, temp_guard):
        """Allow the pipeline to inject the temperature monitor."""
        self.temp_guard = temp_guard
    
    def _verify_connection(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.config.get("model", {}).get('api_request_timeout_seconds', 10))
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"[LLM] Model {self.model} missing. Available: {model_names}")
            else:
                logger.info(f"[LLM] Connected: {self.model}")
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"Ollama verify failed: {e}")
        
    def get_accurate_token_count(self, text: str) -> int:
        """
        Get instantaneous token count using local tiktoken library.
        """
        if not text:
            return 0
        
        if self.tokenizer:
            # This runs in microseconds on the CPU
            return len(self.tokenizer.encode(text))
        
        # Fallback only if tiktoken failed to install/load
        fallback_ratio = self.config.get('chunking', {}).get('fallback_tokens_per_word', 1.5)
        tokens_estimate = int(len(text.split()) * fallback_ratio)
        logger.debug(f"[LLM] Tokenizer unavailable; Estimating token count {tokens_estimate} using fallback ratio {fallback_ratio}")
        return tokens_estimate
        
    def rewrite_chunk(self, chunk_text: str, style_prompt: str, system_prompt: str) -> str:
        """
        Send a chunk to the model for rewriting.
        """
        if not chunk_text or len(chunk_text.strip()) == 0:
            raise ValueError("Chunk text is empty")
        
        # Construct Prompt
        user_message = (
            f"{style_prompt}\n\n"
            f"---TEXT TO REWRITE---\n"
            f"{chunk_text}\n\n"
            f"---REWRITTEN TEXT---"
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "num_thread": self.num_thread,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_predict": self.num_predict,
                "seed": self.seed
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = response.json()
            
            if 'message' not in result or 'content' not in result['message']:
                raise ValueError(f"Unexpected JSON: {result}")
            
            rewritten_text = result['message']['content'].strip()
            
            # Validate output quality
            is_valid, validation_msg = self._validate_response(rewritten_text, chunk_text)
            if not is_valid:
                raise ValidationFailure(f"Validation failed: {validation_msg}", rewritten_text)
            
            logger.debug(f"[LLM] Inference complete: {elapsed:.1f}s")
            return rewritten_text

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama timed out (>{self.timeout}s)")
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Lost connection to {self.base_url}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        except ValueError as e:
            raise ValueError(f"Invalid response: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
        

        
    def rewrite_chunk_with_retry(self, chunk_text: str, style_prompt: str,
                                  system_prompt: str, max_retries: int = 2) -> str:
        """
        Rewrite chunk with exponential backoff and dynamic prompt hardening.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # 1. Prepare dynamic prompt
                current_style_prompt = style_prompt
                
                # If retrying, append strict instructions
                if attempt > 0:
                    target_words = int(len(chunk_text.split()) * self.config.get('processing', {}).get('retry_strictness_ratio', 0.85))
                    
                    stricter_instruction = (self.config.get('prompts', {}).get("additional_instructions_on_retry", 
                        "\n\n[IMPORTANT RETRY INSTRUCTION]: Your previous draft was REJECTED because it was too short."
                        "You MUST NOT summarize. You MUST preserve all dialogue and internal monologue."
                    ))
                    current_style_prompt += stricter_instruction

                    logger.info(f"[LLM] Retry #{attempt} initiated")
                    
                # 2. Execute rewrite
                return self.rewrite_chunk(chunk_text, current_style_prompt, system_prompt)
            
            except (ValidationFailure, TimeoutError, ConnectionError, ValueError) as e:
                last_error = e

                 # Check if this is the final attempt on a validation error
                if isinstance(e, ValidationFailure) and attempt == max_retries:
                    logger.warning(f"[LLM] Max retries reached. Accepting imperfect text (Error: {e}).")
                    return e.text 
                
                if attempt < max_retries:                    
                    logger.warning(f"[LLM] Attempt {attempt + 1} failed: {e}.")
                    
                    # 3. Check Temps before retrying
                    if self.temp_guard:
                        self.temp_guard.check_and_pause()
                    
                else:
                    logger.error(f"[LLM] Failed after {max_retries + 1} attempts")
                    raise

        raise RuntimeError(f"Retries exhausted. Last error: {last_error}")
    
    def _validate_response(self, response_text: str, original_chunk: str) -> Tuple[bool, str]:
        """
        Validate LLM response for quality and completeness.
        """
        
        # Check 1: Empty response
        if not response_text or len(response_text.strip()) == 0:
            return False, "Empty response"
        
        # Check 2: Minimum length (Anti-Summarization)
        original_words = len(original_chunk.split())
        response_words = len(response_text.split())

        min_ratio = self.config.get('processing', {}).get('min_length_ratio', 0.4)
        max_ratio = self.config.get('processing', {}).get('max_length_ratio', 1.5)

        if response_words < original_words * min_ratio:
            msg = f"Too short ({response_words}/{original_words} words)"
            logger.warning(f"[Validation] {msg}")
            return False, msg
        
        # Check 3: Maximum length (Hallucination)
        if response_words > original_words * max_ratio:
            msg = f"Too long ({response_words}/{original_words} words)"
            logger.warning(f"[Validation] {msg}")
            return False, msg
        
        # Check 4: Sentence structure
        sentence_count = (
            response_text.count('.') +
            response_text.count('!') +
            response_text.count('?')
        )
        if sentence_count == 0:
            logger.warning("[Validation] No sentence endings found")
        
        # Check 5: Repetition scan (Informational)
        sentences = [s.strip() for s in re.split(r'[.!?]', response_text) if s.strip()]
        if len(sentences) > self.config.get('processing', {}).get('repetition_threshold_count', 4):
            duplicates = set()
            seen = set()
            for s in sentences:
                if s in seen:
                    duplicates.add(s)
                else:
                    seen.add(s)
            if duplicates:
                logger.warning(f"[Validation] Repetition detected: {list(duplicates)[:3]}")
        
        return True, "OK"