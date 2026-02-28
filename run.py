import sys
from pathlib import Path

# Ensure project root is in sys.path
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from src.deal_yolo_data.app import main
except ImportError as e:
    # Fallback if src is not a package or path issue
    print(f"Error importing app: {e}")
    # Try adding src directly to path
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from deal_yolo_data.app import main

if __name__ == "__main__":
    main()
