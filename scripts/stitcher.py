#!/usr/bin/env python3
"""
Windowed Semantic Pivot Stitching
Fixed: 
1. Replaced O(N^2) string concatenation with O(N) list buffering.
2. Removed unsafe character slicing to prevent mid-word tokenization errors.
3. Reinforced refinement logic to handle window boundary artifacts.
"""

import logging
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ChunkStitcher:
    """
    Implements 'Windowed Semantic Pivot Stitching'.
    Projects text into 384-dimensional vector space to match meaning rather than words.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load params
        stitch_conf = config.get('stitching', {})
        
        self.model_name = stitch_conf.get('model_name', 'all-MiniLM-L6-v2')
        self.window_radius = stitch_conf.get('window_radius', 1)
        self.similarity_threshold = stitch_conf.get('similarity_threshold', 0.65)
        self.scan_depth = stitch_conf.get('scan_depth_sentences', 30)
        self.refine_threshold = stitch_conf.get('refinement_threshold', 0.85)

        logger.info(f"[Stitcher] Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"[Stitcher] Load failed: {e}")
            raise
            

        # 2. Check NLTK tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("[Stitcher] Downloading NLTK 'punkt'...")
            nltk.download('punkt')
            
        # Defaults
        self.window_radius = 1  
        self.similarity_threshold = 0.65 

    def stitch_chapter(self, rewritten_chapter_data: Dict) -> str:
        chunks = rewritten_chapter_data.get('rewritten_chunks', [])
        
        if not chunks:
            logger.warning("[Stitcher] No chunks found")
            return ""
        
        if len(chunks) == 1:
            return chunks[0]['rewritten_text']
        
        # Initialize list with first chunk
        stitched_blocks = [chunks[0]['rewritten_text']]
        
        for i in range(1, len(chunks)):
            curr_chunk_text = chunks[i]['rewritten_text']
            
            # Retrieve last stitched block safely
            prev_block = stitched_blocks[-1]
            
            logger.info(f"[Stitcher] Merging Chunk {i}")
            
            # Execute stitch
            # Returns (Modified_Tail_of_Prev, Head_of_Curr)
            new_prev_tail, new_curr_head = self._stitch_pair_split(prev_block, curr_chunk_text)
            
            # Update blocks
            stitched_blocks[-1] = new_prev_tail  # Trim previous
            stitched_blocks.append(new_curr_head) # Add new
            
        # Join into single string
        return " ".join(stitched_blocks)

    def _stitch_pair_split(self, text_a: str, text_b: str) -> Tuple[str, str]:
        """
        Projects sentences, finds pivot, and returns segments using INDEX SLICING 
        to preserve original whitespace/paragraphs.
        """
        # Load tokenizer for span detection
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        # Get spans (start, end) indices for every sentence
        spans_a = list(tokenizer.span_tokenize(text_a))
        spans_b = list(tokenizer.span_tokenize(text_b))
        
        # Also get text versions for the vectorizer
        sents_a = [text_a[start:end] for start, end in spans_a]
        sents_b = [text_b[start:end] for start, end in spans_b]
        
        if not sents_a: return text_a, text_b
        if not sents_b: return text_a, text_b

        # --- EXISTING VECTOR LOGIC STARTS HERE ---
        offset_a = max(0, len(sents_a) - self.scan_depth)
        scan_sents_a = sents_a[offset_a:]

        limit_b = min(len(sents_b), self.scan_depth)
        scan_sents_b = sents_b[:limit_b]

        vecs_a = self.model.encode(scan_sents_a)
        vecs_b = self.model.encode(scan_sents_b)

        if len(vecs_a) == 0 or len(vecs_b) == 0:
            return text_a, text_b

        smoothed_a = self._apply_sliding_window(vecs_a)
        smoothed_b = self._apply_sliding_window(vecs_b)

        rel_pivot_a, rel_pivot_b, score = self._find_pivot_point(smoothed_a, smoothed_b)
        # --- EXISTING VECTOR LOGIC ENDS HERE ---

        if score >= self.similarity_threshold:
            # Refine
            final_rel_b_idx = self._refine_pivot_alignment(
                vecs_a[rel_pivot_a], 
                vecs_b, 
                rel_pivot_b
            )
            
            # Calculate absolute index of the sentence in the list
            abs_pivot_a_idx = offset_a + rel_pivot_a
            
            logger.info(f"  [Match] A[{abs_pivot_a_idx}] -> B[{final_rel_b_idx}] (Sim: {score:.3f})")
            
            # --- FIX: SLICING INSTEAD OF JOINING ---
            
            # For Chunk A: Cut at the END of the pivot sentence
            # spans_a[i] returns (start, end). We want the 'end' of the pivot sentence.
            cut_point_a = spans_a[abs_pivot_a_idx][1]
            part_a = text_a[:cut_point_a]
            
            # For Chunk B: Cut at the START of the sentence AFTER the pivot
            # We want to skip the matching sentence in B.
            if final_rel_b_idx + 1 < len(spans_b):
                cut_point_b = spans_b[final_rel_b_idx + 1][0]
                part_b = text_b[cut_point_b:]
            else:
                part_b = "" # Pivot was the very last sentence of B
            
            return part_a, part_b
            
        else:
            logger.warning(f"  [Fail] No pivot found (Max: {score:.3f}). Appending.")
            return text_a + "\n\n", text_b

    def _refine_pivot_alignment(self, target_vec_a: np.ndarray, vecs_b: np.ndarray, initial_b_idx: int) -> int:
        """
        Scans a small radius around initial B-pivot using RAW vectors
        to fix boundary smoothing artifacts.
        """
        search_radius = self.window_radius + self.config.get('stitching', {}).get('search_radius_buffer', 2) 
        start = max(0, initial_b_idx - search_radius)
        end = min(len(vecs_b), initial_b_idx + search_radius + 1)
        
        best_idx = initial_b_idx
        best_sim = -1.0
        
        target_reshaped = target_vec_a.reshape(1, -1)
        
        for i in range(start, end):
            # Compare raw vectors
            sim = cosine_similarity(target_reshaped, vecs_b[i].reshape(1, -1))[0][0]
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                
        # Only override if match is very strong
        if best_sim > self.refine_threshold: 
            if best_idx != initial_b_idx:
                logger.debug(f"  [Refine] Shifted pivot B {initial_b_idx}->{best_idx} (Sim: {best_sim:.3f})")
            return best_idx
            
        return initial_b_idx

    def _apply_sliding_window(self, vectors: np.ndarray) -> np.ndarray:
        rows, dims = vectors.shape
        smoothed = np.zeros_like(vectors)
        
        if rows == 0: return smoothed
        
        for i in range(rows):
            start = max(0, i - self.window_radius)
            end = min(rows, i + self.window_radius + 1)
            window_slice = vectors[start:end]
            smoothed[i] = np.mean(window_slice, axis=0)
            
        return smoothed

    def _find_pivot_point(self, vecs_a: np.ndarray, vecs_b: np.ndarray) -> Tuple[int, int, float]:
        if vecs_a.size == 0 or vecs_b.size == 0:
            return 0, 0, 0.0
            
        sim_matrix = cosine_similarity(vecs_a, vecs_b)
        max_idx = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
        return max_idx[0], max_idx[1], float(sim_matrix[max_idx])
    