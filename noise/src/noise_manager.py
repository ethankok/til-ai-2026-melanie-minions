import base64
import io
import os
import torch
from PIL import Image
from advgan import Generator


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[NoiseManager] ignoring invalid {name}={raw!r}; expected float", flush=True)
        return default


def _image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    data = data.view(img.height, img.width, 3)
    return data.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().squeeze(0).clamp(0, 1).mul(255).round().byte().cpu()
    data = tensor.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(data, mode="RGB")


class NoiseManager:
    """Level 9: The AdvGAN Factory Inference"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = _env_float("NOISE_EPSILON", 32.0 / 255.0)
        self.mode = os.environ.get("NOISE_MODE", "detector_stress").strip().lower()
        self.stress_strength = _env_float("NOISE_STRESS_STRENGTH", 1.0)
        self.jpeg_quality = int(_env_float("NOISE_JPEG_QUALITY", 95))
        self.generator_loaded = False

        self.generator = Generator().to(self.device)

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, "advgan_generator.pth")
            self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.generator.eval()
            self.generator_loaded = True
            print(
                f"[NoiseManager] AdvGAN loaded mode={self.mode} "
                f"epsilon={self.epsilon:.4f} stress={self.stress_strength:.2f}",
                flush=True,
            )
        except Exception as e:
            self.generator.eval()
            print(f"[NoiseManager] AdvGAN weights unavailable; using stress-only fallback: {e}", flush=True)

    def _generator_noise(self, original_tensor: torch.Tensor) -> torch.Tensor:
        if not self.generator_loaded:
            return torch.zeros_like(original_tensor)

        raw_noise = self.generator(original_tensor)
        if raw_noise.shape[-2:] != original_tensor.shape[-2:]:
            import torch.nn.functional as F

            raw_noise = F.interpolate(
                raw_noise,
                size=original_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.clamp(raw_noise, -self.epsilon, self.epsilon)

    def _detector_stress_noise(self, original_tensor: torch.Tensor, base_noise: torch.Tensor) -> torch.Tensor:
        _, _, height, width = original_tensor.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )

        # Multiple phases survive common detector strides better than one checkerboard.
        p2 = ((xx + yy) % 2).float().mul(2.0).sub(1.0)
        p4 = (((xx // 2) + (yy // 2)) % 2).float().mul(2.0).sub(1.0)
        p8 = (((xx // 4) - (yy // 4)) % 2).float().mul(2.0).sub(1.0)
        pattern = torch.stack((p2, p4, -p8), dim=0).unsqueeze(0)

        luminance = (
            0.299 * original_tensor[:, 0:1]
            + 0.587 * original_tensor[:, 1:2]
            + 0.114 * original_tensor[:, 2:3]
        )
        grad_x = torch.zeros_like(luminance)
        grad_y = torch.zeros_like(luminance)
        grad_x[..., :, 1:] = (luminance[..., :, 1:] - luminance[..., :, :-1]).abs()
        grad_y[..., 1:, :] = (luminance[..., 1:, :] - luminance[..., :-1, :]).abs()
        edge_gate = torch.clamp((grad_x + grad_y) * 8.0, 0.0, 1.0)
        mask = 0.55 + 0.45 * edge_gate

        stress = pattern * mask * (self.epsilon * self.stress_strength)
        if base_noise.abs().max().item() <= 1e-8:
            return torch.clamp(stress, -self.epsilon, self.epsilon)

        # Let AdvGAN choose signs where it is confident; saturate weak regions with detector stress.
        blended_sign = torch.sign((0.65 * base_noise / max(self.epsilon, 1e-8)) + stress)
        blended_sign = torch.where(blended_sign == 0, torch.sign(stress), blended_sign)
        return torch.clamp(blended_sign * self.epsilon, -self.epsilon, self.epsilon)

    def noise(self, image: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(image)).convert("RGB")
            original_tensor = _image_to_tensor(img, self.device)

            with torch.no_grad():
                base_noise = self._generator_noise(original_tensor)
                if self.mode in ("detector_stress", "level10", "stress"):
                    scaled_noise = self._detector_stress_noise(original_tensor, base_noise)
                else:
                    scaled_noise = base_noise
                adv_tensor = torch.clamp(original_tensor + scaled_noise, 0, 1)

            noised_img = _tensor_to_image(adv_tensor)

            buffered = io.BytesIO()
            noised_img.save(buffered, format="JPEG", quality=self.jpeg_quality)
            return base64.b64encode(buffered.getvalue()).decode("ascii")

        except Exception as exc:
            print(f"AdvGAN Inference error: {exc}")
            return base64.b64encode(image).decode("ascii")
