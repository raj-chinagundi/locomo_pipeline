"""
Simple Answer Evaluator for RAG
Checks if generated answer contains the ground truth.
"""

from typing import Dict


def normalize_text(text) -> str:
    """Normalize text for comparison (lowercase, remove extra spaces/punctuation)."""
    if not text:
        return ""
    # Convert to string first (handles numbers, etc.)
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove punctuation at ends
    text = text.strip('.,!?;:')
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
        return {
            'is_correct': False,
            'has_answer': True
        }
    
    # Check if ground truth is in generated answer
    gen_norm = normalize_text(generated)
    gt_norm = normalize_text(ground_truth)
    is_correct = gt_norm in gen_norm
    
    return {
        'is_correct': is_correct,
        'has_answer': True
    }

