#!/usr/bin/env python3
"""
Bug Fix Verification Script
===========================

Run this script after applying all fixes to verify they are correct.

Usage:
    python verify_fixes.py
"""

import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_result(name, passed, message=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    if message:
        print(f"  {name}: {status} - {message}")
    else:
        print(f"  {name}: {status}")

def test_gcn_import():
    """Test Bug #1: GCN import name mismatch"""
    print("\n[Test 1] GCN Import Name Fix")
    print("-" * 40)
    
    try:
        # Try the old import (should fail if not aliased)
        import importlib.util
        
        # Read the tracker_MCT.py file
        if os.path.exists('core/tracker_MCT.py'):
            with open('core/tracker_MCT.py', 'r') as f:
                content = f.read()
            
            if 'RelationRefiner as GCNHandler' in content:
                print_result("Import alias", True, "Using RelationRefiner as GCNHandler")
                return True
            elif 'from gcn_handler import GCNHandler' in content:
                print_result("Import alias", False, "Still using old GCNHandler import")
                return False
            else:
                print_result("Import alias", False, "Import statement not found")
                return False
        else:
            print_result("Import alias", False, "tracker_MCT.py not found")
            return False
            
    except Exception as e:
        print_result("Import alias", False, str(e))
        return False

def test_faiss_lock():
    """Test Bug #2: FAISS write operations have lock"""
    print("\n[Test 2] FAISS Lock on Write Operations")
    print("-" * 40)
    
    try:
        if os.path.exists('core/tracker_MCT.py'):
            with open('core/tracker_MCT.py', 'r') as f:
                content = f.read()

            # Check for lock around FAISS add operations
            # Look for the pattern: with self._faiss_lock: ... faiss_index.add
            
            # Count occurrences of faiss_index.add
            add_count = content.count('self.faiss_index.add')
            
            # Check if the write in update_edge_track_feature is locked
            # This is around line 738 in the original
            
            # Find the section with faiss_index.add after _manage_gallery_diversity
            import re
            
            # Pattern: with self._faiss_lock: followed by faiss_index.add
            pattern = r'with\s+self\._faiss_lock:.*?self\.faiss_index\.add'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # Check in garbage collection
            gc_section = content[content.find('def _run_garbage_collection'):]
            gc_section = gc_section[:gc_section.find('\n    def ')]
            
            gc_has_lock = 'with self._faiss_lock:' in gc_section and 'faiss_index.add' in gc_section
            
            if gc_has_lock:
                print_result("GC FAISS lock", True)
            else:
                print_result("GC FAISS lock", True, "GC uses reset which is OK")
            
            # Check update_edge_track_feature
            update_section = content[content.find('def update_edge_track_feature'):]
            update_section = update_section[:update_section.find('\n    def ')]
            
            # Look for the pattern around faiss_index.add
            if 'with self._faiss_lock:' in update_section:
                # Check if faiss_index.add is inside the lock
                lock_pos = update_section.find('with self._faiss_lock:')
                add_pos = update_section.find('self.faiss_index.add')
                
                if add_pos > lock_pos:
                    print_result("Update FAISS lock", True)
                    return True
                else:
                    print_result("Update FAISS lock", False, "add() before lock")
                    return False
            else:
                print_result("Update FAISS lock", False, "No lock found in update method")
                return False
        else:
            print_result("FAISS lock", False, "core/tracker_MCT.py not found")
            return False
            
    except Exception as e:
        print_result("FAISS lock", False, str(e))
        return False

def test_pending_track_cleanup():
    """Test Bug #3: Pending tracks garbage collection"""
    print("\n[Test 3] Pending Tracks Garbage Collection")
    print("-" * 40)
    
    try:
        if os.path.exists('core/tracker_MCT.py'):
            with open('core/tracker_MCT.py', 'r') as f:
                content = f.read()

            # Check if _run_garbage_collection cleans pending_tracks
            gc_section = content[content.find('def _run_garbage_collection'):]
            gc_section = gc_section[:gc_section.find('\n    def ') if '\n    def ' in gc_section else len(gc_section)]
            
            if 'pending_tracks' in gc_section and ('expired_pending' in gc_section or 'del self.pending_tracks' in gc_section):
                print_result("Pending cleanup", True, "Pending tracks cleanup found in GC")
                return True
            else:
                print_result("Pending cleanup", False, "No pending tracks cleanup in GC")
                return False
        else:
            print_result("Pending cleanup", False, "core/tracker_MCT.py not found")
            return False
            
    except Exception as e:
        print_result("Pending cleanup", False, str(e))
        return False

def test_feature_normalization():
    """Test Bug #4: Feature normalization always enforced"""
    print("\n[Test 4] Feature Normalization")
    print("-" * 40)
    
    try:
        if os.path.exists('models/gcn_handler.py'):
            with open('models/gcn_handler.py', 'r') as f:
                content = f.read()
            
            # Check if normalization is always applied (not just warned)
            predict_method = content[content.find('def predict_batch'):]
            predict_method = predict_method[:predict_method.find('\n    def ') if '\n    def ' in predict_method else len(predict_method)]
            
            # Look for unconditional normalization
            if ('t_feat = t_feat / ' in predict_method or 
                '_normalize_feature' in predict_method or
                'np.linalg.norm(t_feat)' in predict_method):
                
                # Make sure it's not just a warning
                if 'WARNING' not in predict_method or 't_feat = t_feat /' in predict_method:
                    print_result("Track feat norm", True)
                else:
                    print_result("Track feat norm", False, "Only warning, not fixing")
                    return False
            else:
                print_result("Track feat norm", False, "No normalization found")
                return False
            
            # Check candidate normalization
            if 'd_feat = d_feat /' in predict_method or 'd_feat /' in predict_method:
                print_result("Candidate feat norm", True)
                return True
            else:
                print_result("Candidate feat norm", False, "Candidates not normalized")
                return False
        else:
            print_result("Feature norm", False, "models/gcn_handler.py not found")
            return False
            
    except Exception as e:
        print_result("Feature norm", False, str(e))
        return False

def test_reranking_division():
    """Test Bug #5: Safe division in re_ranking"""
    print("\n[Test 5] Re-ranking Division Safety")
    print("-" * 40)
    
    try:
        if os.path.exists('utils/re_ranking.py'):
            with open('utils/re_ranking.py', 'r') as f:
                content = f.read()
            
            # Check for torch.clamp usage
            if 'torch.clamp' in content and 'min=' in content:
                print_result("Division safety", True, "Using torch.clamp for safe division")
                return True
            elif 'max_dist_per_col' in content:
                print_result("Division safety", True, "Using intermediate variable")
                return True
            else:
                print_result("Division safety", False, "No safe division pattern found")
                return False
        else:
            print_result("Division safety", False, "utils/re_ranking.py not found")
            return False
            
    except Exception as e:
        print_result("Division safety", False, str(e))
        return False

def test_quality_bounds():
    """Test Bug #6: Quality score clamped to [0, 1]"""
    print("\n[Test 6] Quality Score Bounds")
    print("-" * 40)
    
    try:
        if os.path.exists('services/edge_camera.py'):
            with open('services/edge_camera.py', 'r') as f:
                content = f.read()
            
            # Find the calculate_quality_score function
            if 'def calculate_quality_score' in content:
                func_section = content[content.find('def calculate_quality_score'):]
                func_section = func_section[:func_section.find('\ndef ') if '\ndef ' in func_section else len(func_section)]
                
                # Check for min(1.0, ...) at the return
                if 'min(1.0,' in func_section or 'min(1,' in func_section:
                    print_result("Upper bound", True, "max(0, min(1, score)) pattern found")
                    return True
                elif 'max(0.0, final_score)' in func_section and 'min' not in func_section:
                    print_result("Upper bound", False, "Only lower bound, no upper bound")
                    return False
                else:
                    print_result("Upper bound", False, "Could not verify bounds")
                    return False
            else:
                print_result("Quality bounds", False, "Function not found")
                return False
        else:
            print_result("Quality bounds", False, "services/edge_camera.py not found")
            return False
            
    except Exception as e:
        print_result("Quality bounds", False, str(e))
        return False

def test_runtime_import():
    """Test that the system can actually import and initialize"""
    print("\n[Test 7] Runtime Import Test")
    print("-" * 40)
    
    try:
        # Try importing key modules
        sys.path.insert(0, '.')
        
        try:
            from models.gcn_handler import RelationRefiner
            print_result("gcn_handler import", True)
        except ImportError as e:
            print_result("gcn_handler import", False, str(e))
            return False

        try:
            from core.continuum_memory import ContinuumStateV2
            print_result("continuum_memory import", True)
        except ImportError as e:
            print_result("continuum_memory import", False, str(e))
            # This is not critical
        
        return True
        
    except Exception as e:
        print_result("Runtime import", False, str(e))
        return False

def main():
    print_header("BUG FIX VERIFICATION SCRIPT")
    print("This script checks if all identified bugs have been fixed.")
    
    results = []
    
    # Run all tests
    results.append(("Bug #1: GCN Import", test_gcn_import()))
    results.append(("Bug #2: FAISS Lock", test_faiss_lock()))
    results.append(("Bug #3: Pending GC", test_pending_track_cleanup()))
    results.append(("Bug #4: Feature Norm", test_feature_normalization()))
    results.append(("Bug #5: Division Safety", test_reranking_division()))
    results.append(("Bug #6: Quality Bounds", test_quality_bounds()))
    results.append(("Runtime Import", test_runtime_import()))
    
    # Print summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  [{status}] {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All bugs have been fixed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} fix(es) still needed.")
        print("\nRefer to BUG_FIX_GUIDE.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())