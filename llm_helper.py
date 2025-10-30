"""
LLM helper for generating answers from retrieved context.
"""
from typing import List, Dict, Any
import os
import time
import warnings

# Suppress warnings and errors
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress stderr output for ALTS warnings
import sys
import logging
logging.getLogger().setLevel(logging.CRITICAL)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars only

# Rate limiter: 10 requests per minute for Gemini
_last_request_time = None
_request_interval = 6.0  # seconds between requests (60/10 = 6 seconds)


def generate_answer(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Generate an answer to the question using retrieved documents.
    
    Args:
        question: The question to answer
        retrieved_docs: List of retrieved document dictionaries
        
    Returns:
        Generated answer string
    """
    # Build context from retrieved documents with timestamps
    context_parts = []
    for doc in retrieved_docs:
        speaker = doc.get('speaker', 'Speaker')
        text = doc['text']
        timestamp = doc.get('session_timestamp', '')
        
        if timestamp:
            context_parts.append(f"[{timestamp}] {speaker}: {text}")
        else:
            context_parts.append(f"{speaker}: {text}")
    
    context = "\n".join(context_parts)
    
    # Check if Gemini API key is available
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("WARNING: No GEMINI_API_KEY found. Please set it to use Gemini.")
        return "NO_API_KEY"
    
    return generate_answer_with_gemini(question, context, api_key)


def generate_answer_with_gemini(question: str, context: str, api_key: str) -> str:
    """
    Generate answer using Google Gemini API with rate limiting.
    
    Args:
        question: The question to answer
        context: Retrieved context
        api_key: Google API key for Gemini
        
    Returns:
        Generated answer
    """
    global _last_request_time
    
    # Rate limiting: ensure 6 seconds between requests
    if _last_request_time is not None:
        elapsed = time.time() - _last_request_time
        if elapsed < _request_interval:
            sleep_time = _request_interval - elapsed
            time.sleep(sleep_time)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are answering questions based on a conversation. The conversation has timestamps showing when it occurred.

Context (with timestamps):
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- When the context contains relative time references (yesterday, last week, next month, this month, last year, etc.), you MUST calculate the absolute date using the conversation timestamp
  Example: If someone says "next month" on August 28, 2023 → answer "September 2023"
  Example: If someone says "last year" on July 12, 2023 → answer "2022"
  Example: If someone says "yesterday" on May 8, 2023 → answer "May 7, 2023"
- Be specific and precise with dates, numbers, and facts - always return absolute dates, not relative references
- If the information is clearly not in the context, say "Information not found in context"
- Keep your answer brief and factual

Answer:"""
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0,
                'max_output_tokens': 150,
            }
        )
        
        _last_request_time = time.time()
        
        if response.text:
            return response.text.strip()
        else:
            print(f"  WARNING: Empty response from Gemini")
            return "EMPTY_RESPONSE"
        
    except Exception as e:
        print(f"  ERROR with Gemini API: {e}")
        _last_request_time = time.time()
        return f"ERROR: {str(e)[:50]}"



