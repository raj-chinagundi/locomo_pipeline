"""
Simple Answer Evaluator for RAG
Checks if generated answer contains the ground truth.
"""

import re
from typing import Dict


def normalize_text(text) -> str:
    """
    Normalize text for comparison (lowercase, remove extra spaces/punctuation).
    
    This is a more robust version that removes all punctuation,
    not just from the ends.
    """
    if not text:
        return ""
    
    # Convert to string first (handles numbers, etc.)
    text = str(text).lower()
    
    # --- UPDATED LOGIC ---
    # Remove all punctuation characters from anywhere in the string
    # This will turn "7 may, 2023." into "7 may 2023"
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace (e.g. "a   b" -> "a b")
    text = ' '.join(text.split())
    
    return text


def evaluate_answer(generated: str, ground_truth: str) -> Dict:
    """
    Check if generated answer contains the ground truth answer.
    
    Args:
        generated: Generated answer from LLM
        ground_truth: Ground truth answer
    
    Returns:
        Dictionary with:
        - is_correct: bool (whether ground truth is in generated answer)
        - has_answer: bool (whether answer was generated)
    """
    if not generated:
        return {
            'is_correct': False,
            'has_answer': False
        }
    
    if not ground_truth:
        # If there's no ground truth, we can't score it.
        return {
            'is_correct': False,
            'has_answer': True
        }
    
    # Check if ground truth is in generated answer
    gen_norm = normalize_text(generated)
    gt_norm = normalize_text(ground_truth)
    is_correct = gt_norm in gen_norm
    
    # --- DEBUGGING PRINT ---
    # You can uncomment this to see the comparison
    if not is_correct:
        print(f"  [Eval DBG] GT: '{gt_norm}'")
        print(f"  [Eval DBG] GEN: '{gen_norm}'")
    
    return {
        'is_correct': is_correct,
        'has_answer': True
    }
