"""
Quick Smoke Test for AdaStep Integration
Verifies that all components can be imported and initialized correctly.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all AdaStep modules can be imported."""
    print("🔍 Testing imports...")
    try:
        from predictors.adastep import AdaStepAdapter, HorizonPredictor, StateClusterAnalyzer
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_predictor():
    """Test HorizonPredictor initialization and forward pass."""
    print("\n🔍 Testing HorizonPredictor...")
    try:
        from predictors.adastep import HorizonPredictor
        
        predictor = HorizonPredictor(input_dim=512, hidden_dim=256)
        print(f"  Parameters: {predictor.get_num_parameters():,}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 512)
        output = predictor(dummy_input)
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
        assert (output >= 0).all() and (output <= 1).all(), "Output not in [0, 1]"
        
        # Test horizon prediction
        k = predictor.predict_horizon(dummy_input, k_min=5, k_max=50)
        assert k.shape == (4,), f"Expected (4,), got {k.shape}"
        assert (k >= 5).all() and (k <= 50).all(), f"k not in [5, 50]: {k}"
        
        print(f"✅ HorizonPredictor works correctly")
        print(f"  Sample predictions: {k.tolist()}")
        return True
    except Exception as e:
        print(f"❌ HorizonPredictor test failed: {e}")
        return False


def test_analyzer():
    """Test StateClusterAnalyzer."""
    print("\n🔍 Testing StateClusterAnalyzer...")
    try:
        from predictors.adastep import StateClusterAnalyzer
        
        analyzer = StateClusterAnalyzer(num_clusters=5, percentile=50.0)
        
        # Generate dummy data
        np.random.seed(42)
        dummy_states = np.random.randn(100, 512)
        dummy_actions = np.random.randn(100, 50, 14)  # [N, seq_len, action_dim]
        
        # Fit clusters
        analyzer.fit_clusters(dummy_states)
        assert analyzer.kmeans is not None, "KMeans not fitted"
        
        # Test linearity deviation
        deviation = analyzer.calculate_linearity_deviation(dummy_actions[0], k=10)
        assert isinstance(deviation, float), f"Expected float, got {type(deviation)}"
        
        print(f"✅ StateClusterAnalyzer works correctly")
        print(f"  Cluster labels distribution: {np.bincount(analyzer.kmeans.labels_)}")
        return True
    except Exception as e:
        print(f"❌ StateClusterAnalyzer test failed: {e}")
        return False


def test_adapter():
    """Test AdaStepAdapter without policy (fallback mode)."""
    print("\n🔍 Testing AdaStepAdapter...")
    try:
        from predictors.adastep import AdaStepAdapter
        
        # Initialize without policy (uses random features)
        adapter = AdaStepAdapter(
            predictor_ckpt=None,
            policy=None,
            k_min=5,
            k_max=50,
            device='cpu'
        )
        
        # Test prediction with dummy inputs
        dummy_qpos = torch.randn(1, 14)
        dummy_image = torch.randn(1, 3, 480, 640)
        
        k = adapter.predict_horizon(dummy_qpos, dummy_image)
        assert isinstance(k, int), f"Expected int, got {type(k)}"
        assert 5 <= k <= 50, f"k not in [5, 50]: {k}"
        
        # Test statistics
        stats = adapter.get_statistics()
        assert 'mean_k' in stats, "Statistics missing mean_k"
        
        print(f"✅ AdaStepAdapter works correctly")
        print(f"  Predicted horizon: {k}")
        print(f"  Statistics: {stats}")
        return True
    except Exception as e:
        print(f"❌ AdaStepAdapter test failed: {e}")
        return False


def test_integration():
    """Test full integration with mock ACT policy."""
    print("\n🔍 Testing full integration...")
    try:
        from predictors.adastep import AdaStepAdapter, HorizonPredictor
        import torch.nn as nn
        
        # Create mock policy
        class MockPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.encoder = lambda qpos, img: (torch.randn(qpos.shape[0], 512), None)
        
        mock_policy = MockPolicy()
        
        # Initialize adapter with mock policy
        adapter = AdaStepAdapter(
            predictor_ckpt=None,
            policy=mock_policy,
            k_min=5,
            k_max=50,
            device='cpu'
        )
        
        # Run multiple predictions
        for _ in range(10):
            dummy_qpos = torch.randn(1, 14)
            dummy_image = torch.randn(1, 3, 480, 640)
            k = adapter.predict_horizon(dummy_qpos, dummy_image)
            assert 5 <= k <= 50
        
        # Check statistics
        stats = adapter.get_statistics()
        assert stats['total_queries'] == 10
        assert stats['unique_values'] > 0
        
        print(f"✅ Full integration test passed")
        print(f"  Mean k: {stats['mean_k']:.2f}")
        print(f"  Entropy: {stats['entropy']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  AdaStep Integration Smoke Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("HorizonPredictor", test_predictor()))
    results.append(("StateClusterAnalyzer", test_analyzer()))
    results.append(("AdaStepAdapter", test_adapter()))
    results.append(("Full Integration", test_integration()))
    
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<25s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! AdaStep is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
