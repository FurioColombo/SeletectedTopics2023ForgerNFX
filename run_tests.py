"""Quick script to test metrics and capture failures."""
import subprocess
import sys

result = subprocess.run(
    ["pytest", "tests/test_metrics.py", "-v", "--tb=short"],
    capture_output=True,
    text=True
)

print("=" * 80)
print("STDOUT:")
print(result.stdout)
print("=" * 80)
print("STDERR:")
print(result.stderr)
print("=" * 80)
print(f"Return code: {result.returncode}")
