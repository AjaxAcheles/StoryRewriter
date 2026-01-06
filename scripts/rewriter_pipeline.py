#!/usr/bin/env python3
"""
Main orchestration script: loads config, validates, processes chapters, saves output.
Refactored: Wires together TokenAwareChunker (with Ollama dependency) and LCS Stitcher.
"""

import os
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from power_utils import WindowsCpuThrottler
from monitor import TemperatureGuard
from utils import (
    load_config, load_manuscript, load_style_prompt,
    detect_chapters, save_json, save_text
)
from chunker import TokenAwareChunker
from local_llm import OllamaRewriter
from stitcher import ChunkStitcher

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rewriter.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StoryRewriterPipeline:
    """Main pipeline orchestrator with dependency injection and pre-flight checks."""
    
    def __init__(self, config_path: str):
        logger.info(f"[Main] Loading config: {config_path}")
        self.config = load_config(config_path)
        
        # 1. Initialize Rewriter (Chunker dependency)
        self.rewriter = OllamaRewriter(self.config)
        
        # 2. Initialize Chunker
        self.chunker = TokenAwareChunker(self.config, self.rewriter)
        
        # 3. Initialize Stitcher
        self.stitcher = ChunkStitcher(self.config)
        
        # 4. Initialize Hardware Guards
        self.temp_guard = TemperatureGuard(self.config)

        # Inject Guard into Rewriter
        self.rewriter.set_temp_guard(self.temp_guard)
        
        # 5. Load Prompt Files
        self.style_prompt = load_style_prompt()
        self.system_prompt = self.config.get('prompts', {}).get('system_prompt',
            "You are an expert fiction editor. Your task is to improve prose "
            "while preserving all plot, character, perspective, and voice details. "
            "Return ONLY the rewritten text with no commentary or explanation."
        )
        
        self.stats = {
            'total_chapters': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_time': 0,
            'validation_failures': 0
        }
    
    def run(self) -> None:
        """Main execution flow."""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("[Main] STARTING REWRITER PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Run checks
            self._validate_setup()
            
            logger.info(f"[Main] Model: {self.config['model']['model_name']}")
            logger.info(f"[Main] Input: {self.config['manuscript']['input_path']}")
            logger.info("=" * 80)
            
            # Load text
            manuscript = load_manuscript(self.config['manuscript']['input_path'])
            logger.info(f"[Main] Loaded manuscript ({len(manuscript):,} chars)")
            
            # Detect structure
            chapters = detect_chapters(manuscript, self.config)
            self.stats['total_chapters'] = len(chapters)
            logger.info(f"[Main] Detected {len(chapters)} chapters")
            
            # Process with CPU throttling enabled
            with WindowsCpuThrottler():
                processed_chapters = []
                for chapter in chapters:
                    try:
                        processed = self._process_chapter(chapter)
                        processed_chapters.append(processed)
                    except Exception as e:
                        logger.error(f"[Main] Chapter {chapter['number']} Failed: {e}", exc_info=True)
                        if not self.config['processing']['retry_on_failure']:
                            raise
            
            # Output results
            self._save_all_chapters(processed_chapters)
            self._save_metadata(processed_chapters)
            
            elapsed = time.time() - start_time
            self.stats['total_time'] = elapsed
            
            logger.info("=" * 80)
            logger.info(f"[Main] Pipeline Complete: {elapsed/3600:.1f}h")
            logger.info(f"[Main] Stats: {self.stats['successful_chunks']}/{self.stats['total_chunks']} chunks ok")
            if self.stats['validation_failures'] > 0:
                logger.warning(f"[Main] Val Failures: {self.stats['validation_failures']}")
            logger.info("=" * 80)
        
        except KeyboardInterrupt:
            logger.error("[Main] Interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"[Main] CRITICAL FAILURE: {e}", exc_info=True)
            sys.exit(1)
    
    def _validate_setup(self) -> None:
        """Validate configuration, model connection, and file access."""
        logger.info("[Main] Running pre-flight checks...")
        
        # 1. Verify Ollama
        try:
            self.rewriter._verify_connection()
            logger.info("[Main] Ollama check: OK")
        except Exception as e:
            raise RuntimeError(f"Ollama not accessible: {e}")
        
        # 2. Verify Input File
        input_path = Path(self.config['manuscript']['input_path'])
        if not input_path.exists():
            raise FileNotFoundError(f"Manuscript missing: {input_path}")
        logger.info(f"[Main] Input check: OK")
        
        # 3. Check Token Budget
        try:
            system_tokens = self.rewriter.get_accurate_token_count(self.system_prompt)
            style_tokens = self.rewriter.get_accurate_token_count(self.style_prompt)
        except Exception as e:
            logger.warning(f"[Main] Tokenizer fail: {e}. Using estimates.")
            system_tokens = int(len(self.system_prompt.split()) * 1.3)
            style_tokens = int(len(self.style_prompt.split()) * 1.3)
        
        reserved_output = self.config['chunking']['reserve_for_output_tokens']
        context_window = self.config['chunking']['context_window_tokens']
        
        available_for_chunk = context_window - system_tokens - style_tokens - reserved_output
        
        logger.info(
            f"[Main] Token Budget: {available_for_chunk:,} avail "
            f"(Sys:{system_tokens}, Style:{style_tokens}, Res:{reserved_output})"
        )
        
        if available_for_chunk < 800:
            raise ValueError(f"Insufficient tokens ({available_for_chunk}). Increase context window.")
        logger.info("[Main] Budget check: OK")
        
        # 4. Verify Output Dir
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        try:
            test_file.write_text("test", encoding='utf-8')
            test_file.unlink()
            logger.info("[Main] Write check: OK")
        except Exception as e:
            raise PermissionError(f"Cannot write to output: {e}")
        
        logger.info("[Main] All checks passed.\n")
    
    def _process_chapter(self, chapter: Dict) -> Dict:
        """Process a single chapter: chunk, rewrite, stitch."""
        logger.info(f"\n[Main] Processing Ch {chapter['number']}: {chapter['title']}")
        logger.info(f"  Input: {chapter['word_count']:,} words")
        
        # 1. Chunking
        chunks = self.chunker.chunk_chapter(chapter['text'], self.config)
        logger.info(f"  Chunks: {len(chunks)}")

        # Thermal Check
        self.temp_guard.check_and_pause()

        # Process each chunk        
        rewritten_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Log loop progress
                token_count = chunk.get('token_count', 0)
                logger.info(f"    [{i+1}/{len(chunks)}] Rewriting (~{token_count} tokens)...")
                
                start = time.time()
                
                # 2. Rewrite (with retries)
                max_retries = self.config['processing'].get('max_retries', 2)
                
                # Note: chunk['text'] contains Overlap + Content
                rewritten = self.rewriter.rewrite_chunk_with_retry(
                    chunk['text'],
                    self.style_prompt,
                    self.system_prompt,
                    max_retries=max_retries
                )
                
                elapsed = time.time() - start
                
                # 3. Validate
                tolerance = self.config['processing'].get('validation_tolerance_percent', 35) / 100.0
                is_valid, status = self._validate_chunk_rewrite(chunk['text'], rewritten, tolerance=tolerance)
                
                if not is_valid:
                    logger.warning(f"    [WARN] Val Fail: {status}")
                    self.stats['validation_failures'] += 1
                else:
                    logger.info(f"    [OK] {elapsed:.1f}s | {status}")
                
                # Store Data
                rewritten_chunks.append({
                    'index': chunk['index'],
                    'original_text': chunk['text'], 
                    'rewritten_text': rewritten,
                    'token_count': token_count,
                    'processing_time': elapsed,
                    'validation_status': status
                })
                
                self.stats['successful_chunks'] += 1

                # Save debug files if configured
                if self.config['output'].get('save_intermediate_chunks', False):
                    self._save_debug_chunk(chapter, chunk, rewritten)

                # Cooldown between chunks
                if i < len(chunks) - 1:
                    self.temp_guard.check_and_pause()
            
            except Exception as e:
                logger.error(f"    [FAIL] Error: {e}")
                self.stats['failed_chunks'] += 1
                raise
        
        self.stats['total_chunks'] += len(chunks)

        if not rewritten_chunks:
            raise RuntimeError("All chunks failed.")
        
        # 4. Stitching
        stitched_text = self.stitcher.stitch_chapter({'rewritten_chunks': rewritten_chunks})
        logger.info(f"  Stitched: {len(stitched_text.split()):,} words")
        
        return {
            'chapter_number': chapter['number'],
            'chapter_title': chapter['title'],
            'original_text': chapter['text'],
            'rewritten_chunks': rewritten_chunks,
            'stitched_text': stitched_text,
            'processing_time': sum(c['processing_time'] for c in rewritten_chunks)
        }

    def _validate_chunk_rewrite(self, original: str, rewritten: str, tolerance: float) -> tuple:
        """Validate that rewritten chunk stays within length tolerance."""
        original_words = len(original.split())
        rewritten_words = len(rewritten.split())
        
        if original_words == 0:
            return False, "Input empty"
        
        ratio = rewritten_words / original_words
        lower_bound = 1.0 - tolerance
        upper_bound = 1.0 + tolerance
        
        status_msg = f"{rewritten_words}/{original_words} words ({ratio:.2f}x)"
        
        if lower_bound <= ratio <= upper_bound:
            return True, status_msg
        else:
            return False, f"{status_msg} - Out of bounds"

    def _save_debug_chunk(self, chapter, chunk, rewritten):
        """Helper to save debug artifacts."""
        debug_dir = Path(self.config['output']['output_dir']) / 'debug_chunks'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_filename = f"ch{chapter['number']:02d}_chunk{chunk['index']:03d}.md"
        save_path = debug_dir / chunk_filename
        
        debug_content = (
            f"## INPUT (Overlap + Content)\n\n{chunk['text']}\n\n"
            f"## REWRITTEN\n\n{rewritten}\n"
        )
        save_text(debug_content, str(save_path))

    def _save_all_chapters(self, chapters: List[Dict]) -> None:
        """Save individual chapter files and combined manuscript."""
        output_dir = Path(self.config['output']['output_dir'])
        chapters_dir = output_dir / 'chapters'
        chapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chapters
        for ch in chapters:
            ch_file = chapters_dir / f"chapter_{ch['chapter_number']:02d}.md"
            content = f"# {ch['chapter_title']}\n\n{ch['stitched_text']}"
            save_text(content, str(ch_file))
            logger.info(f"  Saved: {ch_file.name}")
        
        # Save Combined
        combined_file = output_dir / 'full_manuscript_rewritten.md'
        combined_content = ""
        for ch in chapters:
            combined_content += f"# {ch['chapter_title']}\n\n{ch['stitched_text']}\n\n---\n\n"
        save_text(combined_content, str(combined_file))
        logger.info(f"  Combined: {combined_file.name}")
    
    def _save_metadata(self, chapters: List[Dict]) -> None:
        """Save processing statistics and metadata."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model': self.config['model']['model_name'],
            'total_processing_hours': self.stats['total_time'] / 3600,
            'stats': self.stats,
            'chapters': []
        }
        
        for ch in chapters:
            metadata['chapters'].append({
                'number': ch['chapter_number'],
                'title': ch['chapter_title'],
                'chunks': len(ch['rewritten_chunks']),
                'original_word_count': len(ch['original_text'].split()),
                'rewritten_word_count': len(ch['stitched_text'].split()),
                'processing_seconds': ch['processing_time']
            })
        
        output_dir = Path(self.config['output']['output_dir'])
        metadata_file = output_dir / 'metadata.json'
        save_json(metadata, str(metadata_file))
        logger.info(f"  Metadata: {metadata_file.name}")

if __name__ == '__main__':
    # CLI Entry Point
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/default.json'
    try:
        pipeline = StoryRewriterPipeline(config_path)
        pipeline.run()
        
        # Optional Shutdown
        if pipeline.config.get('shutdown_after_completion', False):
            logger.info("[Main] Job complete. Shutting down system in 60 seconds...")
            # /s = shutdown, /t 60 = 60 second timer
            os.system(f"shutdown /s /t {pipeline.config.get('system', {}).get('shutdown_timer_seconds', 60)}")
        
    except Exception as e:
        logger.critical(f"[Main] CRITICAL FAILURE: {e}", exc_info=True)
        sys.exit(1)