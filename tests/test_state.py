"""
Unit tests for state.py.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import state

class TestState(unittest.TestCase):

    def setUp(self):
        # Clean any direct attribute bindings set on the state module by other tests
        # to ensure __getattr__ triggers correctly.
        if "active_runs" in state.__dict__:
            del state.__dict__["active_runs"]
        if "active_runs_lock" in state.__dict__:
            del state.__dict__["active_runs_lock"]

    def test_getattr_local_fallback(self):
        # When 'app' module is not loaded
        with patch.dict(sys.modules, {"app": None}):
            active_runs = state.active_runs
            self.assertIsInstance(active_runs, dict)
            
            active_runs_lock = state.active_runs_lock
            self.assertIsNotNone(active_runs_lock)

    def test_getattr_app_loaded_has_attr(self):
        # Mock 'app' module that has active_runs and active_runs_lock
        mock_app = MagicMock()
        mock_app.active_runs = {"run_from_app": True}
        mock_app.active_runs_lock = "lock_from_app"
        
        with patch.dict(sys.modules, {"app": mock_app}):
            self.assertEqual(state.active_runs, {"run_from_app": True})
            self.assertEqual(state.active_runs_lock, "lock_from_app")

    def test_getattr_app_loaded_no_attr(self):
        # Mock 'app' module that does NOT have these attributes
        mock_app = object() # plain object with no dict attributes
        
        with patch.dict(sys.modules, {"app": mock_app}):
            active_runs = state.active_runs
            self.assertIsInstance(active_runs, dict)
            self.assertNotEqual(active_runs, {"run_from_app": True})

    def test_getattr_invalid_name(self):
        with self.assertRaises(AttributeError):
            _ = state.nonexistent_attribute

    def test_get_fn_and_val_app_loaded(self):
        mock_app = MagicMock()
        mock_app.my_func = lambda: "hello"
        mock_app.my_val = 42
        
        with patch.dict(sys.modules, {"app": mock_app}):
            fn = state.get_fn("my_func", None)
            self.assertEqual(fn(), "hello")
            
            val = state.get_val("my_val", None)
            self.assertEqual(val, 42)

    def test_get_fn_and_val_app_not_loaded(self):
        with patch.dict(sys.modules, {"app": None}):
            fn = state.get_fn("my_func", "default_fn")
            self.assertEqual(fn, "default_fn")
            
            val = state.get_val("my_val", "default_val")
            self.assertEqual(val, "default_val")

if __name__ == "__main__":
    unittest.main()
