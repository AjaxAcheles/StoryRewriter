#!/usr/bin/env python3
"""
Token-aware chunking with scene detection and precise measurement.
Refactored to use Ollama's native tokenizer for accuracy.
"""

import re
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class TokenAwareChunker:
    """
    Splits text into chunks respecting scene boundaries and token limits.
    Now requires an OllamaRewriter instance for accurate token counting.
    """
    
    def __init__(self, config, rewriter=None):
        """
        Args:
            config: Configuration dictionary.
            rewriter: Instance of OllamaRewriter (used for token counting).
                      Can be set later via set_rewriter if circular imports are an issue.
        """
        self.config = config
        self.rewriter = rewriter
        
    def _get_token_count(self, text: str) -> int:
        # Return accurate count via API if available, otherwise estimate (1.3 tokens/word)
        if self.rewriter:
            return self.rewriter.get_accurate_token_count(text)
        return int(len(text.split()) * 1.3)

    def _find_sentence_boundary(self, text: str, start_pos: int, direction: str = 'backward') -> int:
        """
        Find the nearest sentence ending (.!?) relative to start_pos.
        """
        # Regex: Punctuation followed by whitespace or quote
        pattern = re.compile(r'[.!?]["\']?\s+')
        
        if direction == 'backward':
            # Scan text preceding start_pos for the last match
            substring = text[:start_pos]
            matches = list(pattern.finditer(substring))
            if matches:
                return matches[-1].end()
            return 0  # Default to start of text
        else:
            # Scan text following start_pos for the first match
            substring = text[start_pos:]
            match = pattern.search(substring)
            if match:
                return start_pos + match.end()
            return len(text)

    def _get_tail_by_tokens(self, text: str, target_tokens: int) -> str:
        """
        Extract roughly the last `target_tokens` from text.
        Efficiently estimates then refines.
        """
        if not text:
            return ""
            
        total_tokens = self._get_token_count(text)
        if total_tokens <= target_tokens:
            return text

        # Step 1: Estimate character count based on target tokens (4 chars/token safety margin)
        estimated_chars = int(target_tokens * self.config.get('chunking', {}).get('chars_per_token_estimate', 4.0))
        search_start = max(0, len(text) - estimated_chars)
        
        # Step 2: Align to the nearest sentence boundary
        snap_pos = self._find_sentence_boundary(text, search_start, direction='forward')
        candidate = text[snap_pos:]
        
        # Step 3: Validate token count
        candidate_tokens = self._get_token_count(candidate)
        
        # Step 4: Iteratively trim if the candidate exceeds the budget by >20%
        while candidate_tokens > target_tokens * 1.2 and len(candidate) > 100:
            next_boundary = self._find_sentence_boundary(candidate, 0, direction='forward')
            if next_boundary == len(candidate) or next_boundary == 0:
                break 
            candidate = candidate[next_boundary:]
            candidate_tokens = self._get_token_count(candidate)
            
        return candidate

    def chunk_chapter(self, chapter_text: str, config: Dict) -> List[Dict]:
        """
        Main entry point.
        1. Detect Scenes
        2. Sub-divide large scenes
        3. Add overlaps
        """
        if not self.rewriter:
            logger.warning("[Chunker] No rewriter provided; using estimates")

        target_tokens = config['chunking']['target_chunk_size_tokens']
        overlap_tokens = config['chunking']['overlap_tokens']
        min_tokens = config['chunking']['min_chunk_size_tokens']
        
        # 1. Split text by visual separators (***, ---, ___, etc.)
        scene_splits = re.split(config.get('chunking', {}).get('scene_separator_regex', r'\\n\\s*[\\*\\-\\_]{3,}\\s*\\n'), chapter_text)
        scenes = [s.strip() for s in scene_splits if s.strip()]
        
        logger.info(f"[Chunker] Scenes detected: {len(scenes)}")
        
        logical_blocks = []
        
        # 2. Process scenes; subdivide if they exceed token limits
        for scene_idx, scene_text in enumerate(scenes):
            scene_tokens = self._get_token_count(scene_text)
            
            if scene_tokens <= target_tokens:
                # Scene fits within one chunk
                logical_blocks.append({
                    'text': scene_text,
                    'tokens': scene_tokens,
                    'is_start_of_scene': True
                })
            else:
                # Scene requires subdivision
                logger.info(f"[Chunker] Subdividing scene {scene_idx+1} (Size: {scene_tokens})")
                sub_blocks = self._split_large_text(scene_text, target_tokens, min_tokens)
                logical_blocks.extend(sub_blocks)

        # 3. Assemble final chunks with context overlap
        final_chunks = []
        current_pos_tracker = 0 
        
        for i, block in enumerate(logical_blocks):
            text_content = block['text']
            
            # Retrieve overlap from the tail of the previous block
            overlap_text = ""
            overlap_len_chars = 0
            
            if i > 0:
                prev_text = logical_blocks[i-1]['text']
                overlap_text = self._get_tail_by_tokens(prev_text, overlap_tokens)
                overlap_len_chars = len(overlap_text)
            
            # Combine overlap and current content for LLM input
            sep = "\n\n" if overlap_text else ""
            full_text = overlap_text + sep + text_content
            
            full_tokens = self._get_token_count(full_text)
            
            # Safety Check: Prevent context window overflow
            context_window = config['chunking']['context_window_tokens']
            if full_tokens > context_window - config.get('chunking', {}).get('safety_buffer_tokens', 500): 
                logger.warning(f"[Chunker] Chunk {i} near limit ({full_tokens}); trimming overlap")
                # Aggressively trim overlap to fit context
                while full_tokens > context_window - config.get('chunking', {}).get('safety_buffer_tokens', 500) and len(overlap_text) > 50:
                    overlap_text = overlap_text[len(overlap_text)//2:]
                    overlap_text = overlap_text[self._find_sentence_boundary(overlap_text, 0, 'forward'):]
                    full_text = overlap_text + "\n\n" + text_content
                    full_tokens = self._get_token_count(full_text)
            
            final_chunks.append({
                'index': i,
                'text': full_text, 
                'content_text': text_content, 
                'overlap_text': overlap_text, 
                'token_count': full_tokens,
                'is_scene_start': block.get('is_start_of_scene', False)
            })
            
        return final_chunks

    def _split_large_text(self, text: str, target_tokens: int, min_tokens: int) -> List[Dict]:
        """
        Recursive/Iterative splitter for text that exceeds chunk limits.
        Splits at sentence boundaries.
        """
        blocks = []
        remaining_text = text
        
        while len(remaining_text) > 0:
            current_tokens = self._get_token_count(remaining_text)
            
            # Base case: Remainder fits in target size
            if current_tokens <= target_tokens:
                blocks.append({
                    'text': remaining_text,
                    'tokens': current_tokens,
                    'is_start_of_scene': (len(blocks) == 0)
                })
                break
                
            # Step 1: Estimate split point (3 chars/token)
            estimated_chars = int(target_tokens * round(self.config.get('chunking', {}).get('chars_per_token_estimate', 4.0)))
            if estimated_chars >= len(remaining_text):
                estimated_chars = len(remaining_text) - 1
                
            # Step 2: Search backward for sentence boundary
            split_pos = self._find_sentence_boundary(remaining_text, estimated_chars, direction='backward')
            
            # Fallback: Search forward if no backward boundary found
            if split_pos == 0:
                 split_pos = self._find_sentence_boundary(remaining_text, estimated_chars, direction='forward')

            # Fallback: Hard split if boundaries are elusive
            if split_pos == 0 or split_pos == len(remaining_text):
                split_pos = min(len(remaining_text), estimated_chars)
            
            chunk_candidate = remaining_text[:split_pos]
            
            # Step 3: Validate measurement
            actual_tokens = self._get_token_count(chunk_candidate)
            
            # Retract if token count exceeds target significantly
            while actual_tokens > target_tokens and split_pos > 100:
                new_pos = self._find_sentence_boundary(chunk_candidate[:-1], len(chunk_candidate)-1, 'backward')
                if new_pos == 0:
                    break 
                split_pos = new_pos
                chunk_candidate = remaining_text[:split_pos]
                actual_tokens = self._get_token_count(chunk_candidate)
            
            # Step 4: Add block and advance
            blocks.append({
                'text': chunk_candidate,
                'tokens': actual_tokens,
                'is_start_of_scene': (len(blocks) == 0)
            })
            
            remaining_text = remaining_text[split_pos:].strip()
            
        return blocks