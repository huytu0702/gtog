import graphrag.api as api
import inspect

print("=" * 80)
print("global_search signature:")
print("=" * 80)
sig = inspect.signature(api.global_search)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

print("\n" + "=" * 80)
print("local_search signature:")
print("=" * 80)
sig = inspect.signature(api.local_search)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

print("\n" + "=" * 80)
print("drift_search signature:")
print("=" * 80)
sig = inspect.signature(api.drift_search)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

print("\n" + "=" * 80)
print("tog_search signature:")
print("=" * 80)
sig = inspect.signature(api.tog_search)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
