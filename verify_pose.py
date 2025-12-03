import sys
import os
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath("backend"))

try:
    from app.ml_core.tools.pose_estimation import PoseEstimator
    from app.ml_core.tools.pipeline import AVAILABLE_TOOL_TYPES
    print("Backend imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Verify Registration
assert 'pose_estimation' in AVAILABLE_TOOL_TYPES, "PoseEstimator not registered in pipeline"

# Verify Initialization (Mocking model load to avoid download in test if possible, or just checking class structure)
# We won't instantiate because it might try to download the model which takes time/network.
# Just checking imports and registration is a good sanity check for now.

print("PoseEstimator registration verified.")
