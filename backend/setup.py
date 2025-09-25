#!/usr/bin/env python3
"""
ReviewSense Backend Setup Script
This script helps set up and run the FastAPI backend properly.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def check_environment_file():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating template...")
        create_env_template()
        return False

    # Check if GEMINI_API_KEY is set
    with open(env_file, 'r') as f:
        content = f.read()

    if 'GEMINI_API_KEY=your_gemini_api_key_here' in content:
        print("⚠️  Please set your GEMINI_API_KEY in the .env file")
        print("   Edit the .env file and replace 'your_gemini_api_key_here' with your actual API key")
        return False

    print("✅ Environment configuration found")
    return True

def create_env_template():
    """Create a template .env file."""
    env_content = """# ReviewSense Backend Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Custom server configuration
# HOST=0.0.0.0
# PORT=8000
# RELOAD=true
"""
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env template file")

def run_server():
    """Run the FastAPI server."""
    print("\n🚀 Starting ReviewSense backend...")
    print("   Server will be available at: http://localhost:8000")
    print("   API documentation: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop the server\n")

    try:
        # Use uvicorn to run the server with auto-reload
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]

        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error running server: {e}")
        print("Try running manually: python main.py")

def main():
    """Main setup function."""
    print("🔧 ReviewSense Backend Setup")
    print("=" * 40)

    # Check system requirements
    check_python_version()

    if not check_pip():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the backend directory")
        sys.exit(1)

    # Install dependencies
    install_dependencies()

    # Check environment
    env_ready = check_environment_file()

    if not env_ready:
        print("\n⚠️  Please configure your .env file before running the server")
        print("   Then run this script again or run: python main.py")
        return

    print("\n" + "=" * 40)
    print("✅ Setup complete! Ready to start the server.")
    print("=" * 40)

    # Ask user if they want to start the server
    response = input("Start the server now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_server()
    else:
        print("Run 'python main.py' when you're ready to start the server")

if __name__ == "__main__":
    main()