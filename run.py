"""Run directly without installing the project or any third-party packages."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from robot_hackathon.cli import main
raise SystemExit(main())
