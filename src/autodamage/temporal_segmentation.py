from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .device import resolve_torch_device

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except Exception:  # pragma: no cover - dépendance optionnelle
    torch = None
    functional = None
    nn = None


if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.SiLU(inplace=True),
            )

        def forward(self, value: "torch.Tensor") -> "torch.Tensor":
            return self.layers(value)


    class SharedEncoder(nn.Module):
        def __init__(self, channels: tuple[int, ...]):
            super().__init__()
            self.stages = nn.ModuleList()
            channels_in = 3
            for channels_out in channels:
                self.stages.append(ConvBlock(channels_in, channels_out))
                channels_in = channels_out

        def forward(self, value: "torch.Tensor") -> list["torch.Tensor"]:
            features = []
            for index, stage in enumerate(self.stages):
                if index:
                    value = functional.max_pool2d(value, 2)
                value = stage(value)
                features.append(value)
            return features


    class TemporalDamageNet(nn.Module):
        """Segmente les différences apprises avec un encodeur partagé avant/après."""

        def __init__(self, channels: tuple[int, ...] = (20, 32, 48, 72, 96)):
            super().__init__()
            if len(channels) != 5:
                raise ValueError("Cinq niveaux de canaux sont requis.")
            self.channels = channels
            self.encoder = SharedEncoder(channels)
            self.bottleneck = ConvBlock(channels[4], channels[4])
            self.decoder = nn.ModuleList(
                ConvBlock(channels[level + 1] + channels[level], channels[level])
                for level in range(3, -1, -1)
            )
            self.head = nn.Conv2d(channels[0], 1, 1)

        def forward(self, before: "torch.Tensor", after: "torch.Tensor") -> "torch.Tensor":
            before_features = self.encoder(before)
            after_features = self.encoder(after)
            differences = [torch.abs(a - b) for a, b in zip(after_features, before_features)]
            value = self.bottleneck(differences[-1])
            for decoder, level in zip(self.decoder, range(3, -1, -1)):
                value = functional.interpolate(value, size=differences[level].shape[-2:], mode="bilinear", align_corners=False)
                value = decoder(torch.cat((value, differences[level]), dim=1))
            return self.head(value)
else:  # pragma: no cover
    class TemporalDamageNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch n'est pas installé.")


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


class OptionalTemporalSegmenter:
    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        tile_size: int = 384,
        overlap: float = 0.25,
    ):
        self.enabled = False
        self.error: str | None = None
        self.device = resolve_torch_device(device)
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = 0.5
        self.minimum_component_area = 0
        self.border_ignore = 0
        self.model = None
        if weights is None:
            return
        if torch is None:
            self.error = "PyTorch indisponible"
            return
        path = Path(weights)
        if not path.exists():
            self.error = f"Poids temporels introuvables: {path}"
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            channels = tuple(checkpoint.get("channels", (20, 32, 48, 72, 96)))
            self.model = TemporalDamageNet(channels=channels)
            self.model.load_state_dict(checkpoint["model_state"])
            self.threshold = float(checkpoint.get("threshold", 0.5))
            self.minimum_component_area = int(checkpoint.get("minimum_component_area", 0))
            self.border_ignore = int(checkpoint.get("border_ignore", 0))
            self.tile_size = int(checkpoint.get("crop_size", tile_size))
            self.model.eval().to(self.device)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - poids externes variables
            self.error = f"Chargement du modèle temporel impossible: {exc}"
            self.model = None

    @staticmethod
    def _tensor(image: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    def predict_probability(self, before: np.ndarray, aligned_after: np.ndarray) -> np.ndarray | None:
        if not self.enabled or self.model is None or torch is None:
            return None
        if before.shape != aligned_after.shape:
            raise ValueError("Le modèle temporel exige deux images alignées de même taille.")
        height, width = before.shape[:2]
        tile = min(self.tile_size, max(height, width))
        padded_height = max(height, tile)
        padded_width = max(width, tile)
        bottom = padded_height - height
        right = padded_width - width
        before_padded = cv2.copyMakeBorder(before, 0, bottom, 0, right, cv2.BORDER_REFLECT_101)
        after_padded = cv2.copyMakeBorder(aligned_after, 0, bottom, 0, right, cv2.BORDER_REFLECT_101)
        accumulator = np.zeros((padded_height, padded_width), np.float32)
        weights = np.zeros_like(accumulator)
        window_1d = np.hanning(tile).astype(np.float32) if tile > 2 else np.ones(tile, np.float32)
        window = np.maximum(np.outer(window_1d, window_1d), 0.05)
        amp_enabled = self.device.startswith("cuda")
        with torch.inference_mode():
            for y in tile_starts(padded_height, tile, self.overlap):
                for x in tile_starts(padded_width, tile, self.overlap):
                    tensor_before = self._tensor(before_padded[y:y + tile, x:x + tile]).to(self.device)
                    tensor_after = self._tensor(after_padded[y:y + tile, x:x + tile]).to(self.device)
                    with torch.autocast(device_type="cuda", enabled=amp_enabled):
                        probability = torch.sigmoid(self.model(tensor_before, tensor_after))[0, 0]
                    probability_np = probability.float().cpu().numpy()
                    accumulator[y:y + tile, x:x + tile] += probability_np * window
                    weights[y:y + tile, x:x + tile] += window
        return (accumulator / np.maximum(weights, 1e-6))[:height, :width]

    def probability_to_mask(self, probability: np.ndarray) -> np.ndarray:
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        if self.border_ignore > 0:
            border = min(self.border_ignore, mask.shape[0] // 2, mask.shape[1] // 2)
            mask[:border] = 0
            mask[-border:] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
        if self.minimum_component_area > 1:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask)
            for label in range(1, count):
                if int(stats[label, cv2.CC_STAT_AREA]) >= self.minimum_component_area:
                    cleaned[labels == label] = 255
            mask = cleaned
        return mask

    def predict_mask(self, before: np.ndarray, aligned_after: np.ndarray) -> tuple[np.ndarray | None, dict]:
        probability = self.predict_probability(before, aligned_after)
        if probability is None:
            return None, {"enabled": False, "error": self.error}
        mask = self.probability_to_mask(probability)
        return mask, {
            "enabled": True,
            "threshold": round(self.threshold, 4),
            "minimum_component_area": self.minimum_component_area,
            "border_ignore": self.border_ignore,
            "peak_probability": round(float(probability.max()), 5),
            "positive_pixel_ratio": round(float(np.mean(mask > 0)), 7),
        }
