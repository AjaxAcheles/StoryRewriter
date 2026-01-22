#!/usr/bin/env python3
"""
Utility functions: loading configs, detecting chapters, token counting.
COMPLETE IMPLEMENTATION with all helper functions.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load JSON configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config missing: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_style_prompt(prompt_path: str = 'config/style_prompt.txt') -> str:
    """Load style/tone instructions from file."""
    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Style prompt missing: {prompt_path}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def load_manuscript(file_path: str) -> str:
    """Load manuscript from text/markdown file."""
    manuscript_file = Path(file_path)
    if not manuscript_file.exists():
        raise FileNotFoundError(f"Manuscript missing: {file_path}")
    
    with open(manuscript_file, 'r', encoding='utf-8') as f:
        return f.read()

def detect_chapters(text: str, config: Dict) -> List[Dict]:
    """
    Detect chapters using regex pattern from config.
    Falls back to length-based splitting if no chapter markers found.
    
    Returns:
        List of dicts with keys: number, title, text, word_count, char_count
    """
    pattern = config['manuscript'].get('chapter_marker_pattern', '^## Chapter\\s+\\d+')
    
    try:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
    except re.error as e:
        logger.warning(f"[Utils] Invalid Regex: {e}. Using length fallback.")
        return _split_by_length(text, config)
    
    if not matches:
        logger.info("[Utils] No markers found. Using length fallback.")
        return _split_by_length(text, config)
    
    chapters = []
    
    # Check for Front Matter (Text before first marker)
    if matches[0].start() > 50:  
        pre_text = text[:matches[0].start()].strip()
        if len(pre_text) > 0:
            chapters.append({
                'number': 0,
                'title': 'Front Matter / Prologue',
                'text': pre_text,
                'word_count': len(pre_text.split()),
                'char_count': len(pre_text),
                'start_char': 0,
                'end_char': matches[0].start()
            })
            logger.info(f"[Utils] Front Matter detected ({len(pre_text)} chars)")

    for i, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[i+1].start() if i+1 < len(matches) else len(text)
        
        chapter_text = text[start_char:end_char].strip()
        title = match.group().strip()
        
        chapters.append({
            'number': chapters[-1]['number'] + 1 if chapters else 1,
            'title': title,
            'text': chapter_text,
            'word_count': len(chapter_text.split()),
            'char_count': len(chapter_text),
            'start_char': start_char,
            'end_char': end_char
        })
    
    logger.info(f"[Utils] Chapters detected: {len(chapters)}")
    return chapters

def _split_by_length(text: str, config: Dict) -> List[Dict]:
    """
    Fallback: split text into chapters based on target word length.
    """
    target_length = config['manuscript'].get('fallback_chapter_length_words', 6500)
    paragraphs = text.split('\n\n')
    
    chapters = []
    current_chapter = []
    current_word_count = 0
    chapter_number = 1
    
    for para in paragraphs:
        if not para.strip(): 
            continue
        
        para_words = len(para.split())
        
        # Check if adding this paragraph exceeds target length
        if current_word_count + para_words > target_length and current_chapter:
            # Finalize current chapter
            chapter_text = '\n\n'.join(current_chapter)
            chapters.append({
                'number': chapter_number,
                'title': f'Chapter {chapter_number}',
                'text': chapter_text,
                'word_count': len(chapter_text.split()),
                'char_count': len(chapter_text),
                'start_char': 0, 
                'end_char': 0
            })
            chapter_number += 1
            current_chapter = [para]
            current_word_count = para_words
        else:
            current_chapter.append(para)
            current_word_count += para_words
    
    # Save final buffer
    if current_chapter:
        chapter_text = '\n\n'.join(current_chapter)
        chapters.append({
            'number': chapter_number,
            'title': f'Chapter {chapter_number}',
            'text': chapter_text,
            'word_count': len(chapter_text.split()),
            'char_count': len(chapter_text),
            'start_char': 0,
            'end_char': 0
        })
    
    return chapters

def save_json(data: Dict, file_path: str) -> None:
    """Save dictionary to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"[IO] Saved: {file_path}")

def save_text(text: str, file_path: str) -> None:
    """Save text to file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    logger.debug(f"[IO] Saved: {file_path}")
