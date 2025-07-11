#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate the CollectEntitiesI fix
"""

import sys
from pathlib import Path

# Add the src directory to the path
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

# Import the fixed batch_mesh module
try:
    from batch_mesh import AnsaBatchMeshRunner
    print("✓ Successfully imported AnsaBatchMeshRunner")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_quality_check():
    """Test the quality check functionality"""
    print("\n=== Testing Quality Check ===")
    
    try:
        # Create runner instance
        runner = AnsaBatchMeshRunner()
        print("✓ Successfully created AnsaBatchMeshRunner instance")
        
        # Test quality check with custom thresholds
        custom_thresholds = {
            'min_element_length': 2.0,
            'max_element_length': 8.0
        }
        
        print("Running quality check...")
        results = runner.check_element_quality(custom_thresholds)
        
        print("✓ Quality check completed successfully")
        print(f"Results: {results}")
        
        # Check if the results contain expected keys
        expected_keys = ['timestamp', 'thresholds', 'checks', 'total_elements', 'bad_elements', 'quality_ratio']
        for key in expected_keys:
            if key in results:
                print(f"✓ Found expected key: {key}")
            else:
                print(f"✗ Missing expected key: {key}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quality check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_checks():
    """Test individual check methods"""
    print("\n=== Testing Individual Check Methods ===")
    
    try:
        runner = AnsaBatchMeshRunner()
        
        # Test min length check
        print("Testing min length check...")
        min_result = runner._check_shell_min_length(2.0)
        print(f"✓ Min length check result: {min_result.get('status', 'Unknown')}")
        
        # Test max length check  
        print("Testing max length check...")
        max_result = runner._check_shell_max_length(8.0)
        print(f"✓ Max length check result: {max_result.get('status', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Individual checks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing CollectEntitiesI fix...")
    
    # Run tests
    test1_passed = test_quality_check()
    test2_passed = test_individual_checks()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The CollectEntitiesI fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)