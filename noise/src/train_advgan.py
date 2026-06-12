import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from advgan import AdvGANTrainer

def main():
    print("1. Waking up the GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    print("2. Securing the Target (Victim) Model...")
    target_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    print("3. Assembling the AdvGAN Factory...")
    trainer = AdvGANTrainer(target_model=target_model, device=device)

    print("4. Establishing Data Pipeline (Loading Imagenette)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./imagenette2/train', transform=transform)

    # 16 keeps T4 16GB VRAM from overflowing at this resolution.
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    epochs = 5

    print("\n=== INITIATING MEXICAN STANDOFF ===")
    for epoch in range(epochs):
        total_G_loss = 0
        total_D_loss = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            loss_G, loss_D = trainer.train_step(images, labels)

            total_G_loss += loss_G
            total_D_loss += loss_D

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] "
                      f"| Generator Loss: {loss_G:.4f} | Detective Loss: {loss_D:.4f}")

    print("\n=== TRAINING COMPLETE ===")
    print("Saving the Generator (The Factory)...")
    torch.save(trainer.generator.state_dict(), "advgan_generator.pth")
    print("Weapon Secured: advgan_generator.pth")

if __name__ == "__main__":
    main()