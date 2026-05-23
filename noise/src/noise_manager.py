import base64
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class NoiseManager:
    """Level 7.1: Overclocked Ghost Attack with Local Telemetry"""

    def __init__(self):
        # --- THE OPTIMIZED BUDGET ---
        self.epsilon = 32.0 / 255.0 
        self.alpha = 4.0 / 255.0    
        self.iters = 20

        self.ensemble = [
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval(),
            models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval(),
            models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT).eval(),
            
            # --- THE TRANSFORMER UPGRADE ---
            # Adding a Vision Transformer to ensure the noise breaks both 
            # convolutional sliding-window math AND transformer patch-attention math.
            models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval()
        ]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.criterion = nn.CrossEntropyLoss()

        self.eot_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

    def calculate_variance_mask(self, image_tensor):
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(image_tensor.device)
        gray = F.conv2d(image_tensor, weights)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(image_tensor.device)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(image_tensor.device)

        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)

        edges = torch.sqrt(edge_x**2 + edge_y**2)
        
        # --- NARROW THE APERTURE ---
        # Reduce the multiplier to stop the noise from spilling into medium textures
        optimized_mask = torch.clamp(edges * 1.5, min=0.0, max=1.0)
        
        return optimized_mask.repeat(1, 3, 1, 1)

    def noise(self, image: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(image)).convert("RGB")
            to_tensor = transforms.ToTensor()
            original_tensor = to_tensor(img).unsqueeze(0)

            norm_orig = self.normalize(self.eot_transform(original_tensor))
            original_preds = []
            with torch.no_grad():
                for model in self.ensemble:
                    outputs = model(norm_orig)
                    _, pred = torch.max(outputs.data, 1)
                    original_preds.append(pred)

            perturbed_tensor = original_tensor.clone()
            spatial_mask = self.calculate_variance_mask(original_tensor)

            for i in range(self.iters):
                perturbed_tensor.requires_grad = True
                eot_tensor = self.eot_transform(perturbed_tensor)
                norm_perturbed = self.normalize(eot_tensor)

                total_loss = 0
                for model_idx, model in enumerate(self.ensemble):
                    outputs = model(norm_perturbed)
                    total_loss += self.criterion(outputs, original_preds[model_idx])

                for model in self.ensemble:
                    model.zero_grad()

                total_loss.backward()
                data_grad = perturbed_tensor.grad.data

                with torch.no_grad():
                    # --- LEVEL 8: MONOCHROME NOISE (ANTI-RAINBOW) ---
                    # 1. Average the gradients across the RGB channels (dim=1)
                    # We squeeze the color dimensions together to find the median light/dark shift.
                    mono_grad = data_grad.mean(dim=1, keepdim=True)
                    
                    # 2. Duplicate that single grayscale gradient back across all 3 channels
                    # This guarantees delta_Red == delta_Green == delta_Blue, preventing hue shifts.
                    color_locked_grad = mono_grad.repeat(1, 3, 1, 1)

                    # 3. Take the step using the color-locked math
                    step = self.alpha * color_locked_grad.sign() * spatial_mask
                    
                    adv_tensor = perturbed_tensor + step
                    eta = torch.clamp(adv_tensor - original_tensor, min=-self.epsilon, max=self.epsilon)
                    perturbed_tensor = torch.clamp(original_tensor + eta, min=0, max=1)

            # --- LOCAL TELEMETRY TRACKER ---
            with torch.no_grad():
                mse = torch.mean((perturbed_tensor - original_tensor) ** 2)
                rmse = torch.sqrt(mse) * 255.0
                print(f"--> [TELEMETRY] Global RMSE: {rmse.item():.4f} (Targeting < 8.0)")

            perturbed_tensor = perturbed_tensor.squeeze(0)
            to_pil = transforms.ToPILImage()
            noised_img = to_pil(perturbed_tensor)

            buffered = io.BytesIO()
            noised_img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("ascii")

        except Exception as exc: 
            print(f"Overclock error: {exc}")
            return base64.b64encode(image).decode("ascii")