import sys
import threading

_local_active_runs = {}
_local_active_runs_lock = threading.Lock()

def __getattr__(name):
    if name == 'active_runs':
        app_mod = sys.modules.get('app')
        if app_mod is not None and hasattr(app_mod, 'active_runs'):
            return app_mod.active_runs
        return _local_active_runs
    if name == 'active_runs_lock':
        app_mod = sys.modules.get('app')
        if app_mod is not None and hasattr(app_mod, 'active_runs_lock'):
            return app_mod.active_runs_lock
        return _local_active_runs_lock
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def get_fn(name, default):
    app_mod = sys.modules.get('app')
    if app_mod is not None:
        return getattr(app_mod, name, default)
    return default

def get_val(name, default):
    app_mod = sys.modules.get('app')
    if app_mod is not None:
        return getattr(app_mod, name, default)
    return default
