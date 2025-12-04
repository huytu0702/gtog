import graphrag.api as api
import inspect

# Check return annotations
print("=" * 80)
print("Return types:")
print("=" * 80)

sig = inspect.signature(api.global_search)
print(f"global_search return: {sig.return_annotation}")

sig = inspect.signature(api.local_search)
print(f"local_search return: {sig.return_annotation}")

sig = inspect.signature(api.drift_search)
print(f"drift_search return: {sig.return_annotation}")

sig = inspect.signature(api.tog_search)
print(f"tog_search return: {sig.return_annotation}")
