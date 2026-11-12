"""
The ROS visualizers publish their images without ``cv_bridge``, whose binary is built
against NumPy 1 and prints an import error under NumPy 2 at every start.
"""

import subprocess
import sys


def test_the_ros_visualizers_load_without_cv_bridge() -> None:
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, robokudo.vis.ros_visualizer; print('cv_bridge' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.strip() == "False"
    assert "NumPy" not in loaded.stderr
