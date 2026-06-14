"""Legacy deployment alias for app.py.

Prefer: streamlit run app.py
This file exists only for older configs that still reference front-end-netflix.py.
"""

from app import main

if __name__ == "__main__":
    main()
