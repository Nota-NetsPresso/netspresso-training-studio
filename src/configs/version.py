from pathlib import Path

# Read API Version from VERSION file
VERSION_FILE = Path(__file__).parent.parent / "VERSION"
__version__ = VERSION_FILE.read_text().strip()

BACKEND_VERSION = __version__
