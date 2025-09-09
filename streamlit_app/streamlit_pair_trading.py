"""
Legacy entry point for Streamlit Cloud deployment
Redirects to the new Home.py in the project root
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import and run the main Home module from project root
try:
    from Home import main
    main()
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import Home module: {e}")
    st.info("Please check that Home.py exists in the project root directory.")