# Test script to identify the exact constructor issue
# Save this as test_constructor.py and run it

from src.core.base_engine import OCRResult, DocumentResult, TextRegion, BoundingBox

# Test 1: Create OCRResult
print("Testing OCRResult creation...")
try:
    ocr_result = OCRResult(
        text="test text",
        confidence=0.8,
        regions=[],
        processing_time=0.1
    )
    print("✓ OCRResult created successfully")
    print(f"  Text: {ocr_result.text}")
    print(f"  Confidence: {ocr_result.confidence}")
except Exception as e:
    print(f"✗ OCRResult failed: {e}")

# Test 2: Create DocumentResult
print("\nTesting DocumentResult creation...")
try:
    doc_result = DocumentResult(
        pages=[ocr_result],
        metadata={'test': 'data'},
        processing_time=0.1,
        engine_name="test",
        confidence_score=0.8
    )
    print("✓ DocumentResult created successfully")
    print(f"  Full text: {doc_result.full_text}")
    print(f"  Confidence: {doc_result.confidence}")
except Exception as e:
    print(f"✗ DocumentResult failed: {e}")

# Test 3: Check what constructor parameters DocumentResult actually expects
print("\nDocumentResult constructor signature:")
import inspect
sig = inspect.signature(DocumentResult.__init__)
print(f"Parameters: {list(sig.parameters.keys())}")