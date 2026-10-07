import json
import sys
from .pipeline import analyze

if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out"), indent=2))
