from __future__ import annotations


def select_device(
    requested: str | None,
    *,
    cuda_available: bool,
    cuda_device_count: int,
) -> str:
    """Valide un périphérique demandé et revient au CPU s'il est inutilisable."""
    value = (requested or "auto").strip().lower()
    if value in {"", "auto"}:
        return "0" if cuda_available and cuda_device_count > 0 else "cpu"
    if value == "cpu":
        return "cpu"

    indices_text = value.removeprefix("cuda:")
    indices = indices_text.split(",")
    if all(index.isdigit() for index in indices):
        valid = cuda_available and all(int(index) < cuda_device_count for index in indices)
        return indices_text if valid else "cpu"
    return "cpu"


def resolve_device(requested: str | None = None) -> str:
    """Sélectionne CUDA uniquement lorsque PyTorch confirme qu'il est disponible."""
    try:
        import torch

        return select_device(
            requested,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()),
        )
    except Exception:
        return "cpu"


def resolve_torch_device(requested: str | None = None) -> str:
    """Retourne la notation ``cuda:N`` attendue par ``torch.nn.Module.to``."""
    selected = resolve_device(requested)
    if selected == "cpu":
        return selected
    return f"cuda:{selected.split(',')[0]}"
