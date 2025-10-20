def run_diagnostics():
    import socket
    import shutil
    import os
    print("=== 🔍 Flask Diagnostic Check ===")

    # Resolve project root (assume scripts/ lives under PROJECT_ROOT/scripts)
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    # Check Flask import
    try:
        import flask
        try:
            ver = flask.__version__
        except Exception:
            # flask 3.2 may deprecate __version__
            from importlib.metadata import version
            ver = version('Flask')
        print("✅ Flask version:", ver)
    except ImportError:
        print("❌ Flask not installed. Run: pip install flask")

    # Check key folders (relative to project root)
    print("\n📁 Folder check:")
    for folder in ['templates', 'static', 'models', 'data']:
        p = os.path.join(PROJECT_ROOT, folder)
        if os.path.exists(p):
            print(f"✅ {folder}/ found.")
        else:
            print(f"⚠️  Missing folder: {folder}/")

    # Check model file
    model_path = os.path.join(PROJECT_ROOT, 'models', 'xgboost_kc_house.pkl')
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"⚠️  Model file missing: {model_path}")

    # Check if port 5000 is free
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('127.0.0.1', 5000))
        if result == 0:
            print("⚠️  Port 5000 is already in use — try another port (e.g., 8080).")
        else:
            print("✅ Port 5000 available.")
    except Exception as e:
        print("⚠️  Port check failed:", e)
    finally:
        sock.close()

    # Check if index.html exists
    html_path = os.path.join(PROJECT_ROOT, 'real_estate_portal', 'templates', 'index.html')
    if os.path.exists(html_path):
        print("✅ index.html found.")
    else:
        print("⚠️  Missing: templates/index.html")

    # Check Python version
    import sys
    print("\n🐍 Python version:", sys.version)
    print("=== End of diagnostics ===\n")


if __name__ == '__main__':
    run_diagnostics()
