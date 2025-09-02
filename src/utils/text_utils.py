import re
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities for OCR results"""
    
    def __init__(self):
        pass
    
    # Note: analyze_text_quality and _classify_text_type have been moved outside this class
    # to allow them to be imported as standalone functions by other modules.


def _classify_text_type(text: str) -> str:
    """Classify the type of text content"""
    text_lower = text.lower()
    
    # Financial documents
    if re.search(r'\b(invoice|receipt|bill|total|amount|price|\$|€|£|\d+\.\d{2})\b', text_lower):
        return 'Financial Document'
    
    # Forms
    elif re.search(r'\b(name|address|phone|email|form)\b', text_lower):
        return 'Form/Personal Information'
    
    # Short text (signs, labels)
    elif len(text.split()) <= 5:
        return 'Sign/Label'
    
    # Default
    else:
        return 'General Text'


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Analyze text content and return characteristics"""
    if not text or not text.strip():
        return {
            'word_count': 0,
            'character_count': 0,
            'contains_numbers': False,
            'contains_special_characters': False,
            'text_type': 'Empty',
            'estimated_language': 'Unknown'
        }
    
    # Basic statistics
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # Content analysis
    has_numbers = bool(re.search(r'\d', text))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', text))
    
    # Text type classification
    text_type = _classify_text_type(text)
    
    return {
        'word_count': word_count,
        'character_count': char_count,
        'contains_numbers': has_numbers,
        'contains_special_characters': has_special,
        'text_type': text_type,
        'estimated_language': 'English'
    }


def clean_text(text: str) -> str:
    """Clean and normalize OCR text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Fix common OCR errors
    replacements = {
        '0': 'O',  # Sometimes zeros are mistaken for O
        'l': 'I',  # lowercase l for uppercase I
    }
    
    # Apply replacements cautiously
    # (You might want to make this more sophisticated)
    
    return cleaned.strip()

def correct_ocr_errors(text: str) -> str:
    """
    Corrects common OCR errors such as character misrecognition and spacing issues.
    
    Args:
        text (str): The raw text from OCR.
        
    Returns:
        str: The corrected and cleaned text.
    """
    if not text:
        return ""
    
    # Step 1: Basic cleaning using the existing function
    cleaned_text = clean_text(text)
    
    # Step 2: More advanced error correction using regex
    # Fix common misrecognitions (e.g., '1' for 'I', 'S' for '$')
    cleaned_text = re.sub(r'\b(l|1)\b', 'I', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(\d)\s*(?=\d)', r'\1', cleaned_text)  # Remove spaces between numbers
    cleaned_text = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', cleaned_text)  # Fix hyphenated words
    
    # Fix multiple periods or commas
    cleaned_text = re.sub(r'\.+', '.', cleaned_text)
    cleaned_text = re.sub(r',+', ',', cleaned_text)
    
    return cleaned_text

def extract_confidence_score(result: Dict) -> float:
    """Extract confidence score from OCR result"""
    return result.get('confidence', 0.0)

def extract_structured_data(text: str, text_type: str) -> Dict[str, Any]:
    """
    Extracts structured key-value pairs from text based on its classified type.
    
    Args:
        text (str): The raw text from OCR.
        text_type (str): The type of text, e.g., 'Financial Document', 'Form/Personal Information'.
    
    Returns:
        Dict[str, Any]: A dictionary of extracted data.
    """
    extracted_data = {}
    
    if text_type == 'Financial Document':
        # Simple regex for common financial fields
        invoice_match = re.search(r'\b(invoice|bill)\s*[:#]?\s*(\S+)', text, re.IGNORECASE)
        if invoice_match:
            extracted_data['invoice_id'] = invoice_match.group(2)
        
        total_match = re.search(r'\b(total|amount|price)\s*[:]?\s*([$€£]?\s*[\d,]+\.?\d{0,2})', text, re.IGNORECASE)
        if total_match:
            extracted_data['total_amount'] = total_match.group(2)
            
    elif text_type == 'Form/Personal Information':
        # Simple regex for personal info fields
        name_match = re.search(r'\b(name)\s*[:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text, re.IGNORECASE)
        if name_match:
            extracted_data['name'] = name_match.group(2)
            
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        if email_match:
            extracted_data['email'] = email_match.group(0)
            
    return extracted_data


def analyze_and_consolidate(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes and consolidates results from multiple OCR engines.
    
    Args:
        all_results (Dict[str, Any]): A dictionary of results from all engines.
    
    Returns:
        Dict[str, Any]: The consolidated and analyzed result.
    """
    best_result = None
    max_confidence = -1
    
    for engine, result in all_results.items():
        if result.get('status') == 'SUCCESS':
            confidence = extract_confidence_score(result)
            if confidence > max_confidence:
                max_confidence = confidence
                best_result = result
                best_result['source'] = engine
                
    if not best_result:
        return {"error": "No successful OCR results to analyze."}

    # Perform text analysis on the best result
    text_info = analyze_text_quality(best_result.get('text', ''))
    
    # Perform structured data extraction based on text type
    structured_data = extract_structured_data(best_result.get('text', ''), text_info['text_type'])
    
    return {
        "best_result": best_result,
        "text_info": text_info,
        "structured_data": structured_data,
        "all_results": all_results
    }


def get_engine_comparison(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a performance comparison report for all OCR engines.
    
    Args:
        all_results (Dict[str, Any]): The raw results from all OCR engines.
        
    Returns:
        Dict[str, Any]: A dictionary with a summary of each engine's performance.
    """
    comparison_report = {}
    for engine, result in all_results.items():
        engine_summary = {
            "status": result.get("status", "UNKNOWN"),
            "confidence": result.get("confidence", 0.0),
            "processing_time_ms": result.get("processing_time", 0.0),
            "text_length": result.get("text_length", 0),
            "text_preview": result.get("text", "")[:100] + "..." if result.get("text") else ""
        }
        comparison_report[engine] = engine_summary
    return comparison_report

def perform_text_analysis(text: str) -> Dict[str, Any]:
    """
    Orchestrates the full text analysis pipeline for a given text string.
    
    This function cleans, corrects, analyzes, and extracts structured data from the text.
    
    Args:
        text (str): The raw text from an OCR engine.
        
    Returns:
        Dict[str, Any]: A dictionary containing all processed text information.
    """
    # 1. Correct common OCR errors
    corrected_text = correct_ocr_errors(text)
    
    # 2. Analyze the quality and type of the corrected text
    analysis_results = analyze_text_quality(corrected_text)
    text_type = analysis_results.get('text_type', 'General Text')
    
    # 3. Extract structured data based on the identified text type
    structured_data = extract_structured_data(corrected_text, text_type)
    
    # 4. Return the consolidated results
    return {
        "cleaned_text": corrected_text,
        "analysis": analysis_results,
        "structured_data": structured_data
    }
