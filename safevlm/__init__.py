"""Minimal LLaVA baseline and SAPT inference."""

__all__ = ["LlavaInference"]


def __getattr__(name):
    # Data conversion can import safevlm.mmsafety without installing GPU dependencies.
    if name == "LlavaInference":
        from .inference import LlavaInference
        return LlavaInference
    raise AttributeError(name)
