import os
import sys


def _ensure_paths_on_syspath():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(project_root, "src")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_paths_on_syspath()


