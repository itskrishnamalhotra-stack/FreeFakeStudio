"""Validate the actual vision quantizer without downloading or loading a model."""
import importlib
from importlib import metadata
import json
import sys
import traceback


REQUIREMENTS = (
    "transformers>=5.13,<6",
    "accelerate>=1.1,<2",
    "bitsandbytes>=0.46.1,<1",
    "sentencepiece",
)


def diagnostics():
    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        packages[name] = {"version": version, "path": getattr(sys.modules.get(name), "__file__", None)}
    return {"python": sys.executable, "packages": packages}


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


def validate():
    from packaging.requirements import Requirement

    for value in REQUIREMENTS:
        requirement = Requirement(value)
        version = metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"{requirement.name} {version} does not satisfy {value}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Avatar image analysis needs a CUDA GPU. Select a Colab/Kaggle GPU runtime.")
    # Import the same model adapter and validator used by from_pretrained.
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    Bnb4BitHfQuantizer(quantization_config()).validate_environment(device_map={"": 0})
    functional = importlib.import_module("bitsandbytes.functional")
    with torch.inference_mode():
        sample = torch.linspace(-1, 1, 256, device="cuda:0", dtype=torch.float16).reshape(16, 16)
        packed, state = functional.quantize_4bit(sample, quant_type="nf4")
        restored = functional.dequantize_4bit(packed, state)
        if restored.shape != sample.shape or not torch.isfinite(restored).all().item():
            raise RuntimeError("bitsandbytes CUDA NF4 round-trip returned invalid values")
        torch.cuda.synchronize()
    return {"ok": True, "backend": "CUDA NF4", **diagnostics()}


def main():
    try:
        report = validate()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc(), **diagnostics()}), flush=True)
        raise SystemExit(1)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
