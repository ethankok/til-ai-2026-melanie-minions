import base64
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
# We import the Generator architecture you built earlier!
from advgan import Generator 

class NoiseManager:
    """Level 9: The AdvGAN Factory Inference"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 32.0 / 255.0
        
        # 1. Instantiate the empty architecture
        self.generator = Generator().to(self.device)
        
        # 2. Load the trained brain dynamically
        try:
            import os
            # This armor ensures the server ALWAYS finds the weights, 
            # no matter what weird path the Docker container runs from.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, "advgan_generator.pth")
            
            self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.generator.eval()
            print("[+] AdvGAN Payload Loaded Successfully.")
        except Exception as e:
            print(f"[-] Missing advgan_generator.pth! Error: {e}")

        # The Competition limits
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def noise(self, image: bytes) -> str:
        try:
            # 1. Load the target image
            img = Image.open(io.BytesIO(image)).convert("RGB")
            original_tensor = self.to_tensor(img).unsqueeze(0).to(self.device)

            # 2. THE SINGLE FORWARD PASS
            # We don't need a 20-step loop anymore. The Generator instantly knows
            # exactly what noise pattern will break this specific image structure.
            # 2. THE SINGLE FORWARD PASS
            with torch.no_grad():
                raw_noise = self.generator(original_tensor)
                
                # --- THE DIMENSIONAL FIX ---
                # Force the noise mask to perfectly match the original image dimensions
                import torch.nn.functional as F
                raw_noise = F.interpolate(
                    raw_noise, 
                    size=(original_tensor.size(2), original_tensor.size(3)), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                scaled_noise = torch.clamp(raw_noise, -self.epsilon, self.epsilon)
                adv_tensor = torch.clamp(original_tensor + scaled_noise, 0, 1)

            # 3. Ship it
            adv_tensor = adv_tensor.squeeze(0).cpu()
            noised_img = self.to_pil(adv_tensor)

            buffered = io.BytesIO()
            noised_img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("ascii")

        except Exception as exc: 
            print(f"AdvGAN Inference error: {exc}")
            return base64.b64encode(image).decode("ascii")