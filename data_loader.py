"""
Data loader for loading and filtering conversation data.
"""
import json
from typing import List, Dict, Any


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load conversation data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of conversation dictionaries
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def filter_by_category(qa_pairs: List[Dict[str, Any]], category: int) -> List[Dict[str, Any]]:
    """
    Filter QA pairs by category.
    
    Args:
        qa_pairs: List of QA pair dictionaries
        category: Category number to filter by
        
    Returns:
        Filtered list of QA pairs matching the category
    """
    return [qa for qa in qa_pairs if qa.get("category") == category]

