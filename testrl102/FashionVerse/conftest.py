"""
FashionVerse — pytest configuration
Ensures the project root is in sys.path so all imports resolve correctly.
"""

import sys
import os

# Add the FashionVerse directory (project root) to path
sys.path.insert(0, os.path.dirname(__file__))
