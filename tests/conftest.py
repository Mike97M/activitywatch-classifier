"""Point the app at a throwaway data dir and model before any test imports it."""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="aw-classifier-tests-")
os.environ["AW_CLASSIFIER_DATA_DIR"] = _tmp
os.environ["AW_CLASSIFIER_MODEL_PATH"] = os.path.join(_tmp, "fast_classifier.joblib")
