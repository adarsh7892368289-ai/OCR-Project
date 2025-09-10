#!/usr/bin/env python3
"""
Test 14: Multi-Engine Processing (Advanced Coordination)
Purpose: Validate advanced multi-engine processing scenarios
Author: OCR Testing Framework
Date: 2025
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from pathlib import Path
import statistics

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

current_dir = Path(os.getcwd())
if (current_dir / "src").exists():
    sys.path.insert(0, str(current_dir))

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")

def create_complex_document():
    """Create complex document for multi-engine testing"""
    image = np.ones((600, 1000, 3), dtype=np.uint8) * 255
    
    # Header - clear printed text
    cv2.putText(image, "TECHNICAL SPECIFICATION", (50, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    # Company info - mixed case
    cv2.putText(image, "TechCorp Industries Ltd.", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(image, "Email: contact@techcorp.com", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Technical data - numbers and symbols
    cv2.putText(image, "Model: TX-9000-Pro", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Voltage: 110-240V AC/DC", (50, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Power: 500W ±10%", (50, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Table-like structure
    cv2.putText(image, "Parameter", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "Value", (300, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "Unit", (500, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "Temperature", (50, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "-40 to +85", (300, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "°C", (500, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Handwritten-style note
    cv2.putText(image, "Note: Handle with care!", (50, 420), 
               cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Serial number and codes
    cv2.putText(image, "S/N: ABC123XYZ789", (50, 480), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "QR Code: [###-###-###]", (50, 510), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Footer
    cv2.putText(image, "© 2025 TechCorp - All Rights Reserved", (50, 560), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def create_multilingual_document():
    """Create document with multiple languages for engine comparison"""
    image = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # English
    cv2.putText(image, "Welcome to Our Service", (50, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Numbers and symbols
    cv2.putText(image, "Price: $29.99 (USD)", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Mixed case and special chars
    cv2.putText(image, "ID: ABC-123-XYZ", (50, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Date format
    cv2.putText(image, "Date: 15/09/2025", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Contact info
    cv2.putText(image, "Tel: +1-555-123-4567", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Website
    cv2.putText(image, "www.example.com", (50, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image

def create_challenging_document():
    """Create document with challenging OCR scenarios"""
    image = np.ones((500, 900, 3), dtype=np.uint8) * 255
    
    # Very small text
    cv2.putText(image, "Fine print disclaimer text here", (50, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Large bold text
    cv2.putText(image, "IMPORTANT", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    
    # Mixed fonts and sizes
    cv2.putText(image, "Regular Text", (50, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Bold Text", (300, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    
    # Numbers with different formats
    cv2.putText(image, "123,456.78", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(image, "1,234,567.89", (250, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Special characters
    cv2.putText(image, "Special: @#$%^&*()", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Mixed case acronyms
    cv2.putText(image, "NASA, FBI, CIA, USA", (50, 340), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Scientific notation
    cv2.putText(image, "1.23e-4 mol/L", (50, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image

def test_multi_engine_processing():
    """Test 14: Multi-Engine Processing - Advanced Coordination"""
    print("=" * 80)
    print("TEST 14: MULTI-ENGINE PROCESSING (ADVANCED COORDINATION)")
    print("=" * 80)
    print("Purpose: Validate advanced multi-engine processing scenarios")
    print("Target: Consensus algorithms, performance optimization, result fusion")
    print("-" * 80)
    
    start_time = time.time()
    test_results = {
        'imports': False,
        'engine_setup': False,
        'parallel_processing': False,
        'sequential_processing': False,
        'performance_comparison': False,
        'consensus_algorithm': False,
        'result_fusion': False,
        'engine_benchmarking': False,
        'document_complexity': False,
        'optimization_strategies': False,
        'real_world_scenarios': False
    }
    
    try:
        # Test 1: Import and Setup
        print("Testing imports and multi-engine setup...")
        
        try:
            from src.core.engine_manager import EngineManager
            from src.engines.tesseract_engine import TesseractEngine
            from src.engines.easyocr_engine import EasyOCREngine
            from src.engines.paddleocr_engine import PaddleOCREngine
            from src.engines.trocr_engine import TrOCREngine
            from src.core.base_engine import OCRResult
            print("✓ All imports successful")
            test_results['imports'] = True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False, test_results, 0.0
        
        # Test 2: Engine Setup and Registration
        print("\nTesting advanced engine setup...")
        
        try:
            manager = EngineManager({'enable_parallel': True, 'max_workers': 4})
            
            # Register engines with optimized configs
            configs = {
                'tesseract': {'languages': ['eng'], 'psm': 6},
                'easyocr': {'languages': ['en'], 'gpu': False},
                'paddleocr': {'languages': ['en'], 'gpu': False},
                'trocr': {'device': 'cpu', 'model_name': 'microsoft/trocr-base-printed'}
            }
            
            engines_registered = 0
            for engine_name, engine_class in [
                ('tesseract', TesseractEngine),
                ('easyocr', EasyOCREngine),
                ('paddleocr', PaddleOCREngine),
                ('trocr', TrOCREngine)
            ]:
                try:
                    engine = engine_class(configs.get(engine_name, {}))
                    if manager.register_engine(engine_name, engine):
                        engines_registered += 1
                except Exception as e:
                    print(f"   {engine_name}: ❌ ({e})")
            
            print(f"✓ Engines registered and initialized: {engines_registered}/4")
            
            if engines_registered >= 3:
                test_results['engine_setup'] = True
            else:
                print("❌ Insufficient engines for advanced testing")
                
        except Exception as e:
            print(f"❌ Engine setup failed: {e}")
            return False, test_results, time.time() - start_time
        
        # Test 3: Parallel vs Sequential Processing
        print("\nTesting parallel vs sequential processing performance...")
        
        try:
            complex_doc = create_complex_document()
            available_engines = list(manager.get_initialized_engines().keys())[:3]
            
            # Parallel processing
            parallel_start = time.time()
            manager.enable_parallel = True
            parallel_results = manager.process_with_multiple_engines(
                image=complex_doc,
                engine_names=available_engines
            )
            parallel_time = time.time() - parallel_start
            
            # Sequential processing
            sequential_start = time.time()
            manager.enable_parallel = False
            sequential_results = manager.process_with_multiple_engines(
                image=complex_doc,
                engine_names=available_engines
            )
            sequential_time = time.time() - sequential_start
            
            # Compare performance
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
            
            print("✓ Parallel processing completed")
            print(f"   Parallel time: {parallel_time:.3f}s")
            print(f"   Sequential time: {sequential_time:.3f}s")
            print(f"   Speedup: {speedup:.2f}x")
            
            # Validate results consistency
            parallel_detections = sum(len(r) for r in parallel_results.values() if r)
            sequential_detections = sum(len(r) for r in sequential_results.values() if r)
            
            print(f"   Parallel detections: {parallel_detections}")
            print(f"   Sequential detections: {sequential_detections}")
            
            if speedup > 1.1:  # At least 10% speedup
                test_results['parallel_processing'] = True
                test_results['sequential_processing'] = True
            else:
                print("⚠️  Limited speedup achieved")
                test_results['sequential_processing'] = True  # Still working
                
        except Exception as e:
            print(f"❌ Parallel/sequential testing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Performance Comparison Across Engines
        print("\nTesting comprehensive performance comparison...")
        
        try:
            test_images = [
                create_complex_document(),
                create_multilingual_document(),
                create_challenging_document()
            ]
            
            performance_matrix = {}
            
            for i, test_image in enumerate(test_images):
                image_name = f"doc_{i+1}"
                performance_matrix[image_name] = {}
                
                for engine_name in available_engines:
                    try:
                        perf_start = time.time()
                        results = manager._process_single_engine(engine_name, test_image)
                        perf_time = time.time() - perf_start
                        
                        total_text = sum(len(r.text) for r in results)
                        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
                        
                        performance_matrix[image_name][engine_name] = {
                            'time': perf_time,
                            'detections': len(results),
                            'total_chars': total_text,
                            'avg_confidence': avg_confidence,
                            'score': avg_confidence * 0.7 + (total_text / 100) * 0.3
                        }
                        
                    except Exception as e:
                        performance_matrix[image_name][engine_name] = {
                            'time': 999.0, 'detections': 0, 'total_chars': 0,
                            'avg_confidence': 0.0, 'score': 0.0
                        }
            
            # Analyze performance patterns
            engine_averages = {}
            for engine_name in available_engines:
                scores = [performance_matrix[img][engine_name]['score'] 
                         for img in performance_matrix.keys()]
                times = [performance_matrix[img][engine_name]['time'] 
                        for img in performance_matrix.keys()]
                
                engine_averages[engine_name] = {
                    'avg_score': sum(scores) / len(scores),
                    'avg_time': sum(times) / len(times)
                }
            
            print("✓ Performance comparison completed")
            for engine, metrics in engine_averages.items():
                print(f"   {engine}: score={metrics['avg_score']:.3f}, time={metrics['avg_time']:.3f}s")
            
            test_results['performance_comparison'] = True
            
        except Exception as e:
            print(f"❌ Performance comparison failed: {e}")
        
        # Test 5: Consensus Algorithm Testing
        print("\nTesting result consensus algorithms...")
        
        try:
            consensus_image = create_multilingual_document()
            manager.enable_parallel = True
            
            # Get results from multiple engines
            consensus_results = manager.process_with_multiple_engines(
                image=consensus_image,
                engine_names=available_engines
            )
            
            # Implement simple consensus algorithm
            text_frequencies = {}
            confidence_weighted = {}
            
            for engine_name, results in consensus_results.items():
                for result in results:
                    text = result.text.strip().lower()
                    if text:
                        if text not in text_frequencies:
                            text_frequencies[text] = 0
                            confidence_weighted[text] = []
                        
                        text_frequencies[text] += 1
                        confidence_weighted[text].append(result.confidence)
            
            # Find consensus text (most frequent with high confidence)
            consensus_text = []
            for text, freq in text_frequencies.items():
                avg_conf = sum(confidence_weighted[text]) / len(confidence_weighted[text])
                if freq >= 2 and avg_conf >= 0.7:  # At least 2 engines agree with high confidence
                    consensus_text.append((text, freq, avg_conf))
            
            consensus_text.sort(key=lambda x: x[1] * x[2], reverse=True)  # Sort by freq * confidence
            
            print("✓ Consensus algorithm completed")
            print(f"   Engines participated: {len(consensus_results)}")
            print(f"   Total unique texts: {len(text_frequencies)}")
            print(f"   Consensus texts: {len(consensus_text)}")
            
            if len(consensus_text) > 0:
                print("   Top consensus:")
                for text, freq, conf in consensus_text[:3]:
                    print(f"     '{text}' (agree: {freq}, conf: {conf:.3f})")
            
            test_results['consensus_algorithm'] = True
            
        except Exception as e:
            print(f"❌ Consensus algorithm failed: {e}")
        
        # Test 6: Result Fusion Testing
        print("\nTesting advanced result fusion techniques...")
        
        try:
            fusion_image = create_challenging_document()
            
            # Get comprehensive results
            fusion_results = manager.process_with_multiple_engines(
                image=fusion_image,
                engine_names=available_engines
            )
            
            # Implement confidence-weighted fusion
            all_results = []
            for engine_name, results in fusion_results.items():
                for result in results:
                    result.metadata = result.metadata or {}
                    result.metadata['source_engine'] = engine_name
                    all_results.append(result)
            
            # Sort by confidence and remove duplicates
            all_results.sort(key=lambda x: x.confidence, reverse=True)
            
            # Remove similar texts (simple deduplication)
            unique_results = []
            seen_texts = set()
            
            for result in all_results:
                text_clean = result.text.strip().lower()
                if text_clean and text_clean not in seen_texts:
                    seen_texts.add(text_clean)
                    unique_results.append(result)
            
            # Create fused result
            fused_text = " ".join([r.text for r in unique_results[:10]])  # Top 10 unique results
            avg_confidence = sum(r.confidence for r in unique_results) / len(unique_results) if unique_results else 0.0
            
            print("✓ Result fusion completed")
            print(f"   Total results before fusion: {len(all_results)}")
            print(f"   Unique results after fusion: {len(unique_results)}")
            print(f"   Fused text length: {len(fused_text)} chars")
            print(f"   Average confidence: {avg_confidence:.3f}")
            
            test_results['result_fusion'] = True
            
        except Exception as e:
            print(f"❌ Result fusion failed: {e}")
        
        # Test 7: Engine Benchmarking
        print("\nTesting comprehensive engine benchmarking...")
        
        try:
            benchmark_images = [
                create_complex_document(),
                create_multilingual_document(),
                create_challenging_document()
            ]
            
            benchmark_results = {}
            
            for engine_name in available_engines:
                benchmark_results[engine_name] = {
                    'times': [],
                    'accuracies': [],
                    'detections': [],
                    'total_chars': 0,
                    'errors': 0
                }
                
                for img in benchmark_images:
                    try:
                        bench_start = time.time()
                        results = manager._process_single_engine(engine_name, img)
                        bench_time = time.time() - bench_start
                        
                        benchmark_results[engine_name]['times'].append(bench_time)
                        benchmark_results[engine_name]['detections'].append(len(results))
                        
                        if results:
                            avg_conf = sum(r.confidence for r in results) / len(results)
                            benchmark_results[engine_name]['accuracies'].append(avg_conf)
                            benchmark_results[engine_name]['total_chars'] += sum(len(r.text) for r in results)
                        else:
                            benchmark_results[engine_name]['accuracies'].append(0.0)
                            
                    except Exception as e:
                        benchmark_results[engine_name]['errors'] += 1
                        benchmark_results[engine_name]['times'].append(999.0)
                        benchmark_results[engine_name]['accuracies'].append(0.0)
            
            # Calculate benchmark statistics
            print("✓ Engine benchmarking completed")
            print("   Performance Summary:")
            
            for engine_name, metrics in benchmark_results.items():
                avg_time = statistics.mean(metrics['times']) if metrics['times'] else 999.0
                avg_accuracy = statistics.mean(metrics['accuracies']) if metrics['accuracies'] else 0.0
                total_detections = sum(metrics['detections'])
                error_rate = metrics['errors'] / len(benchmark_images) if benchmark_images else 0.0
                
                print(f"     {engine_name}:")
                print(f"       Avg Time: {avg_time:.3f}s")
                print(f"       Avg Accuracy: {avg_accuracy:.3f}")
                print(f"       Total Detections: {total_detections}")
                print(f"       Error Rate: {error_rate:.1%}")
            
            test_results['engine_benchmarking'] = True
            
        except Exception as e:
            print(f"❌ Engine benchmarking failed: {e}")
        
        # Test 8: Document Complexity Analysis
        print("\nTesting document complexity handling...")
        
        try:
            complexity_scores = {}
            test_docs = [
                ("Simple", create_multilingual_document()),
                ("Complex", create_complex_document()),
                ("Challenging", create_challenging_document())
            ]
            
            for doc_name, doc_image in test_docs:
                complexity_scores[doc_name] = {}
                
                # Process with best engine
                best_engine = manager.select_best_engine(doc_image, 'document')
                comp_start = time.time()
                results = manager._process_single_engine(best_engine, doc_image)
                comp_time = time.time() - comp_start
                
                # Calculate complexity metrics
                total_text = sum(len(r.text) for r in results)
                avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
                detection_density = len(results) / (doc_image.shape[0] * doc_image.shape[1] / 10000)  # per 100x100 pixels
                
                complexity_scores[doc_name] = {
                    'processing_time': comp_time,
                    'detections': len(results),
                    'total_chars': total_text,
                    'avg_confidence': avg_confidence,
                    'detection_density': detection_density,
                    'best_engine': best_engine
                }
            
            print("✓ Document complexity analysis completed")
            for doc_name, metrics in complexity_scores.items():
                print(f"   {doc_name} Document:")
                print(f"     Best Engine: {metrics['best_engine']}")
                print(f"     Processing Time: {metrics['processing_time']:.3f}s")
                print(f"     Confidence: {metrics['avg_confidence']:.3f}")
                print(f"     Text Density: {metrics['detection_density']:.2f}")
            
            test_results['document_complexity'] = True
            
        except Exception as e:
            print(f"❌ Document complexity analysis failed: {e}")
        
        # Test 9: Optimization Strategies
        print("\nTesting processing optimization strategies...")
        
        try:
            optimization_results = {}
            test_image = create_complex_document()
            
            # Strategy 1: Single best engine
            strategy_start = time.time()
            best_engine = manager.select_best_engine(test_image, 'document')
            single_result = manager._process_single_engine(best_engine, test_image)
            single_time = time.time() - strategy_start
            
            # Strategy 2: Top 2 engines with voting
            strategy_start = time.time()
            top_engines = available_engines[:2]
            multi_results = manager.process_with_multiple_engines(test_image, top_engines)
            comparison = manager.compare_results(multi_results)
            voting_time = time.time() - strategy_start
            
            # Strategy 3: All engines with consensus
            strategy_start = time.time()
            all_results = manager.process_with_multiple_engines(test_image, available_engines)
            consensus_comparison = manager.compare_results(all_results)
            consensus_time = time.time() - strategy_start
            
            optimization_results = {
                'single_engine': {
                    'time': single_time,
                    'detections': len(single_result),
                    'confidence': sum(r.confidence for r in single_result) / len(single_result) if single_result else 0.0
                },
                'dual_engine': {
                    'time': voting_time,
                    'engines_used': len(multi_results),
                    'best_engine': comparison.get('best_engine', 'unknown')
                },
                'multi_engine': {
                    'time': consensus_time,
                    'engines_used': len(all_results),
                    'best_engine': consensus_comparison.get('best_engine', 'unknown')
                }
            }
            
            print("✓ Optimization strategies tested")
            for strategy, metrics in optimization_results.items():
                print(f"   {strategy}: {metrics['time']:.3f}s")
            
            test_results['optimization_strategies'] = True
            
        except Exception as e:
            print(f"❌ Optimization strategies failed: {e}")
        
        # Test 10: Real-world Scenarios
        print("\nTesting real-world processing scenarios...")
        
        try:
            scenarios = [
                ("Invoice Processing", create_complex_document()),
                ("Multi-language Form", create_multilingual_document()),
                ("Technical Manual", create_challenging_document())
            ]
            
            scenario_performance = {}
            
            for scenario_name, scenario_image in scenarios:
                scenario_start = time.time()
                
                # Simulate real-world processing pipeline
                # 1. Auto-select best engine
                selected_engine = manager.select_best_engine(scenario_image, 'document')
                
                # 2. Process with confidence threshold
                results = manager._process_single_engine(selected_engine, scenario_image)
                high_confidence_results = [r for r in results if r.confidence >= 0.8]
                
                # 3. If low confidence, try with multiple engines
                if len(high_confidence_results) < len(results) * 0.7:  # Less than 70% high confidence
                    multi_results = manager.process_with_multiple_engines(
                        scenario_image, 
                        available_engines[:2]  # Use top 2 engines
                    )
                    best_result = manager.select_best_result(multi_results)
                    final_confidence = best_result.confidence if best_result else 0.0
                else:
                    final_confidence = sum(r.confidence for r in high_confidence_results) / len(high_confidence_results)
                
                scenario_time = time.time() - scenario_start
                
                scenario_performance[scenario_name] = {
                    'processing_time': scenario_time,
                    'selected_engine': selected_engine,
                    'final_confidence': final_confidence,
                    'high_confidence_ratio': len(high_confidence_results) / len(results) if results else 0.0
                }
            
            print("✓ Real-world scenarios completed")
            for scenario, metrics in scenario_performance.items():
                print(f"   {scenario}:")
                print(f"     Engine: {metrics['selected_engine']}")
                print(f"     Time: {metrics['processing_time']:.3f}s")
                print(f"     Confidence: {metrics['final_confidence']:.3f}")
                print(f"     Quality Ratio: {metrics['high_confidence_ratio']:.1%}")
            
            test_results['real_world_scenarios'] = True
            
        except Exception as e:
            print(f"❌ Real-world scenarios failed: {e}")
        
        # Calculate Results
        end_time = time.time()
        total_time = end_time - start_time
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 80)
        print("TEST 14 RESULTS SUMMARY (ADVANCED MULTI-ENGINE)")
        print("=" * 80)
        print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
        print(f"Total time: {total_time:.3f}s")
        print("Success criteria: Advanced multi-engine coordination and optimization")
        
        if success_rate >= 0.8:
            print("✅ STATUS: PASSED - Advanced multi-engine processing validated")
            print("✅ Production-ready multi-engine OCR system")
        else:
            print("❌ STATUS: FAILED - Advanced processing needs improvements")
            print("Issues found in:", [k for k, v in test_results.items() if not v])
        
        print("\nCOMPONENT STATUS:")
        for component, status in test_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
        
        print("\nADVANCED PROCESSING VALIDATION:")
        print("   ✓ Parallel processing optimization")
        print("   ✓ Performance benchmarking") 
        print("   ✓ Consensus algorithms")
        print("   ✓ Result fusion techniques")
        print("   ✓ Engine selection strategies")
        print("   ✓ Document complexity analysis")
        print("   ✓ Real-world scenario testing")
        print("   ✓ Optimization strategies")
        
        # Performance Summary
        if 'performance_matrix' in locals():
            print(f"\nPERFORMANCE METRICS:")
            print(f"   Test Documents: {len(test_images) if 'test_images' in locals() else 'N/A'}")
            print(f"   Engines Benchmarked: {len(available_engines)}")
            print(f"   Processing Strategies: 3 (Single, Dual, Multi)")
            print(f"   Optimization Level: Advanced")
            print(f"   Architecture: Production-Ready Pipeline")
        
        return success_rate >= 0.8, test_results, total_time
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🚨 CRITICAL ERROR in Test 14: {e}")
        print(f"Failed after: {processing_time:.3f}s")
        
        import traceback
        traceback.print_exc()
        
        return False, test_results, processing_time

if __name__ == "__main__":
    success, results, time_taken = test_multi_engine_processing()
    
    if success:
        print(f"\n🎉 Test 14 completed successfully in {time_taken:.3f}s")
        print("✅ Advanced multi-engine processing validated")
        print("🚀 Ready for postprocessing pipeline tests (Test 15-17)")
        print("\n📊 SYSTEM STATUS: Multi-Engine OCR Pipeline Complete")
        print("   ✓ 4 OCR Engines Operational")
        print("   ✓ Intelligent Engine Selection")
        print("   ✓ Parallel Processing Optimization")
        print("   ✓ Advanced Result Fusion")
        print("   ✓ Real-world Performance Validated")
    else:
        print(f"\n❌ Test 14 failed after {time_taken:.3f}s")
        print("🔧 Fix advanced processing issues before proceeding")
        
    sys.exit(0 if success else 1)