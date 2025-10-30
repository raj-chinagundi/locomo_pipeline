"""
Answer evaluator for checking correctness of generated answers.
"""


def evaluate_answer(source_answer: str, generated_answer: str) -> str:
    """
    Evaluate if the generated answer contains the ground truth.
    
    Args:
        source_answer: The ground truth answer
        generated_answer: The generated answer from the model
        
    Returns:
        "correct" or "incorrect"
    """
    # Skip evaluation for error cases
    if generated_answer.startswith("ERROR") or generated_answer in ["NO_API_KEY", "EMPTY_RESPONSE"]:
        return "incorrect"
    
    # Normalize both answers
    source_norm = normalize_answer(source_answer)
    generated_norm = normalize_answer(generated_answer)
    
    # Check for exact match
    if source_norm == generated_norm:
        return "correct"
    
    # Check if ground truth is present in generated answer
    if source_norm in generated_norm:
        return "correct"
    
    # Check for numeric equivalence
    if is_numeric_match(source_norm, generated_norm):
        return "correct"
    
    # Check for date variations (e.g., "7 May 2023" vs "May 7 2023")
    if has_common_date_elements(source_norm, generated_norm):
        return "correct"
    
    return "incorrect"


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    Args:
        answer: The answer string
        
    Returns:
        Normalized answer
    """
    # Convert to lowercase
    answer = str(answer).lower().strip()
    
    # Convert number words to digits
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }
    for word, digit in number_words.items():
        answer = answer.replace(word, digit)
    
    # Remove common punctuation
    for char in [",", ".", "!", "?", ":", ";", "'", '"']:
        answer = answer.replace(char, "")
    
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    return answer


def is_numeric_match(answer1: str, answer2: str) -> bool:
    """
    Check if two answers are numerically equivalent.
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        True if numerically equivalent, False otherwise
    """
    try:
        # Extract numbers from both answers
        num1 = extract_number(answer1)
        num2 = extract_number(answer2)
        
        if num1 is not None and num2 is not None:
            return abs(num1 - num2) < 0.01
    except:
        pass
    
    return False


def extract_number(text: str) -> float:
    """
    Extract a number from text.
    
    Args:
        text: Text containing a number
        
    Returns:
        Extracted number or None
    """
    import re
    
    # Try to find a number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])
    
    return None


def has_common_date_elements(answer1: str, answer2: str) -> bool:
    """
    Check if two answers share common date elements (for date matching).
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        True if they share significant date components
    """
    import re
    
    # Extract numbers (days, years)
    nums1 = set(re.findall(r'\d+', answer1))
    nums2 = set(re.findall(r'\d+', answer2))
    
    # Extract month names
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december',
              'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    months1 = set([m for m in months if m in answer1])
    months2 = set([m for m in months if m in answer2])
    
    # If they share at least one number and one month, consider it a match
    if nums1 and nums2 and months1 and months2:
        if len(nums1 & nums2) > 0 and len(months1 & months2) > 0:
            return True
    
    return False

