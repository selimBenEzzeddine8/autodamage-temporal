from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .device import resolve_torch_device

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - dépendance optionnelle
    torch = None
    nn = None


if nn is not None:
    class TinySiamese(nn.Module):
        """Petit vérificateur siamois entraînable sur paires de patchs 96x96."""

        def __init__(self, embedding_dim: int = 96):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, embedding_dim), nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 3, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 1)
            )

        def forward(self, before: "torch.Tensor", after: "torch.Tensor") -> "torch.Tensor":
            eb = self.encoder(before)
            ea = self.encoder(after)
            features = torch.cat([eb, ea, torch.abs(eb - ea)], dim=1)
            return self.classifier(features).squeeze(1)
else:  # pragma: no cover
    class TinySiamese:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch n'est pas installé. Installez requirements-ml.txt.")


class SiameseVerifier:
    def __init__(self, weights: str | Path | None = None, device: str = "cpu"):
        self.enabled = False
        self.device = resolve_torch_device(device)
        self.model = None
        self.error: str | None = None
        if weights is None:
            return
        if torch is None:
            self.error = "PyTorch indisponible"
            return
        path = Path(weights)
        if not path.exists():
            self.error = f"Poids siamois introuvables: {path}"
            return
        self.model = TinySiamese()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.enabled = True

    @staticmethod
    def _tensor(patch: np.ndarray):
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.25
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, before_patch: np.ndarray, after_patch: np.ndarray) -> float | None:
        if not self.enabled or self.model is None or torch is None:
            return None
        with torch.no_grad():
            logit = self.model(self._tensor(before_patch).to(self.device), self._tensor(after_patch).to(self.device))
            return float(torch.sigmoid(logit).item())
