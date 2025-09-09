"""
Streamlit Cloud entry point - imports and runs Home.py
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the main Home module
from Home import main

if __name__ == "__main__":
    main()