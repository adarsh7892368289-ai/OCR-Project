"""
Test Runner - Runs all pipeline tests in sequence and reports results
Usage: python run_all_tests.py
"""
import sys
import os
from pathlib import Path
import subprocess
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_test_file(test_file, description):
    """Run a specific test file and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING: {description}")
    print(f"   File: {test_file}")
    print('='*60)
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        duration = time.time() - start_time
        
        print(f"\n⏱️  Duration: {duration:.2f}s")
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            # Count passed tests
            passed = result.stdout.count(' PASSED')
            failed = result.stdout.count(' FAILED')
            errors = result.stdout.count(' ERROR')
            
            return {
                'status': 'PASSED',
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
        else:
            print(f"❌ {description} - FAILED")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            
            # Count tests even for failures
            passed = result.stdout.count(' PASSED')
            failed = result.stdout.count(' FAILED')
            errors = result.stdout.count(' ERROR')
            
            return {
                'status': 'FAILED',
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except Exception as e:
        print(f"💥 {description} - ERROR")
        print(f"Error running test: {e}")
        return {
            'status': 'ERROR',
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'duration': 0,
            'output': '',
            'error': str(e)
        }

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking Dependencies...")
    
    required_modules = [
        'pytest',
        'PIL', 
        'numpy'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install pytest pillow numpy")
        return False
    
    print("✅ All dependencies available")
    return True

def main():
    """Run all test suites"""
    print("🚀 Advanced OCR Library - Test Suite Runner")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Define test suites in order
    test_suites = [
        {
            'file': 'test_preprocessing_pipeline.py',
            'name': 'Preprocessing Pipeline Tests',
            'description': 'Tests image_processor.py orchestration + quality_analyzer.py + text_detector.py'
        },
        {
            'file': 'test_engine_coordination_pipeline.py', 
            'name': 'Engine Coordination Tests',
            'description': 'Tests engine_coordinator.py + content_classifier.py + individual engines'
        },
        {
            'file': 'test_postprocessing_pipeline.py',
            'name': 'Postprocessing Pipeline Tests', 
            'description': 'Tests text_processor.py + result_fusion.py + confidence_analyzer.py + layout_reconstructor.py'
        },
        {
            'file': 'test_full_integration_pipeline.py',
            'name': 'Full Integration Tests',
            'description': 'Tests complete OCR pipeline integration through core.py and main API'
        }
    ]
    
    # Run each test suite
    results = {}
    total_start = time.time()
    
    for suite in test_suites:
        test_file = suite['file']
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            results[suite['name']] = {
                'status': 'MISSING',
                'passed': 0,
                'failed': 0, 
                'errors': 1,
                'duration': 0
            }
            continue
            
        results[suite['name']] = run_test_file(test_file, suite['name'])
    
    total_duration = time.time() - total_start
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL TEST RESULTS SUMMARY")
    print('='*80)
    
    total_passed = 0
    total_failed = 0
    total_errors = 0
    suite_results = []
    
    for suite_name, result in results.items():
        status_icon = {
            'PASSED': '✅',
            'FAILED': '❌', 
            'ERROR': '💥',
            'MISSING': '⚠️'
        }.get(result['status'], '❓')
        
        print(f"{status_icon} {suite_name}")
        print(f"   Passed: {result['passed']}, Failed: {result['failed']}, Errors: {result['errors']}")
        print(f"   Duration: {result['duration']:.2f}s")
        
        total_passed += result['passed']
        total_failed += result['failed'] 
        total_errors += result['errors']
        
        suite_results.append({
            'name': suite_name,
            'status': result['status'],
            'stats': f"{result['passed']}/{result['failed']}/{result['errors']}"
        })
        print()
    
    # Overall statistics
    total_tests = total_passed + total_failed + total_errors
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📈 OVERALL STATISTICS")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "   No tests run")
    print(f"   Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "")
    print(f"   Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)" if total_tests > 0 else "")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    # Pipeline health assessment
    print(f"\n🏥 PIPELINE HEALTH ASSESSMENT")
    
    pipeline_health = {
        'Preprocessing Pipeline Tests': results.get('Preprocessing Pipeline Tests', {}).get('status'),
        'Engine Coordination Tests': results.get('Engine Coordination Tests', {}).get('status'),  
        'Postprocessing Pipeline Tests': results.get('Postprocessing Pipeline Tests', {}).get('status'),
        'Full Integration Tests': results.get('Full Integration Tests', {}).get('status')
    }
    
    healthy_pipelines = sum(1 for status in pipeline_health.values() if status == 'PASSED')
    
    if healthy_pipelines == 4:
        print("🎉 ALL PIPELINES HEALTHY - Ready for production!")
    elif healthy_pipelines == 3:
        print("⚠️  3/4 pipelines healthy - Minor issues detected")
    elif healthy_pipelines == 2:
        print("🔧 2/4 pipelines healthy - Moderate issues need attention")
    elif healthy_pipelines == 1:
        print("🚨 1/4 pipelines healthy - Major issues need fixing")
    else:
        print("💀 NO PIPELINES HEALTHY - System needs significant work")
    
    # Critical issue detection
    print(f"\n🔍 CRITICAL ISSUE DETECTION")
    
    critical_issues = []
    
    # Check for specific performance issues
    for suite_name, result in results.items():
        if result['status'] != 'PASSED' and 'output' in result:
            output = result['output'].lower()
            
            # Check for known critical issues
            if 'too many text regions' in output or '2660' in output:
                critical_issues.append("Text Detection Bug: Excessive regions detected (2660 → should be < 100)")
            
            if 'very little text extracted' in output or '9 character' in output:
                critical_issues.append("TrOCR Performance Bug: Minimal text extraction (9 chars → should be 100s+)")
            
            if 'preprocessing may have failed' in output:
                critical_issues.append("Preprocessing Pipeline: Integration failure")
                
            if 'engines may have failed' in output:
                critical_issues.append("Engine Coordination: Integration failure")
                
            if 'postprocessing failed' in output:
                critical_issues.append("Postprocessing Pipeline: Integration failure")
    
    if critical_issues:
        print("🚨 Critical Issues Found:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No critical issues detected")
    
    # Next steps recommendation
    print(f"\n🎯 NEXT STEPS RECOMMENDATIONS")
    
    if healthy_pipelines == 4:
        print("1. ✅ All systems operational - Ready for deployment")
        print("2. 🚀 Consider performance optimization and advanced features")
        print("3. 📚 Update documentation and create user guides")
    elif healthy_pipelines >= 2:
        print("1. 🔧 Fix failing pipeline components first")
        print("2. ✅ Validate fixes with focused unit tests") 
        print("3. 🔄 Re-run full test suite to verify")
    else:
        print("1. 🏗️  Focus on basic component implementation")
        print("2. 📋 Review architecture and dependencies")
        print("3. ⚡ Implement missing core functionality")
    
    # Exit with appropriate code
    if total_failed > 0 or total_errors > 0:
        print(f"\n❌ Tests failed. Exit code: 1")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed. Exit code: 0") 
        sys.exit(0)

if __name__ == "__main__":
    main()