import importlib.util
import os
import sys


PACKAGE_NAME = "_freefakestudio_comfyui_gguf"


def load_gguf_node_mappings(comfyui_root):
    """Import ComfyUI-GGUF as a package so its relative imports work."""
    package_dir = os.path.join(comfyui_root, "custom_nodes", "ComfyUI-GGUF")
    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.isfile(init_path):
        raise RuntimeError(
            "ComfyUI-GGUF is missing. Run the notebook cell with UPDATE_APP=True once."
        )

    existing = sys.modules.get(PACKAGE_NAME)
    if existing is not None:
        mappings = getattr(existing, "NODE_CLASS_MAPPINGS", {})
        if mappings:
            return mappings

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        init_path,
        submodule_search_locations=[package_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create a package loader for ComfyUI-GGUF: {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == PACKAGE_NAME or name.startswith(PACKAGE_NAME + "."):
                sys.modules.pop(name, None)
        raise

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    if "UnetLoaderGGUF" not in mappings:
        raise RuntimeError("ComfyUI-GGUF did not register UnetLoaderGGUF.")
    return mappings
