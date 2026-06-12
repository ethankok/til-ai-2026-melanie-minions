import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    A lightweight Autoencoder. It takes a 3-channel image, compresses it
    to understand the structure, and expands it back into a 3-channel noise mask.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.enc1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))

        # tanh bounds output to [-1, 1] as a noise mask, scaled later by epsilon.
        noise = torch.tanh(self.dec2(x))
        return noise


class Discriminator(nn.Module):
    """
    A binary classifier. It looks at an image and outputs a single number:
    1.0 = Clean (Real)
    0.0 = Poisoned (Fake)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        self.fc = nn.Linear(32 * 28 * 28, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)

        validity = torch.sigmoid(self.fc(x))
        return validity


class AdvGANTrainer:
    def __init__(self, target_model, device):
        self.device = device
        self.target_model = target_model.to(self.device).eval()

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # TTUR: discriminator LR > generator LR
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.adversarial_loss = nn.BCELoss()

        self.epsilon = 32.0 / 255.0

    def train_step(self, clean_images, true_labels):
        # --- Discriminator step ---
        self.optimizer_D.zero_grad()

        raw_noise = self.generator(clean_images)
        scaled_noise = torch.clamp(raw_noise, -self.epsilon, self.epsilon)
        poisoned_images = torch.clamp(clean_images + scaled_noise, 0, 1)

        real_guess = self.discriminator(clean_images)
        loss_D_real = self.adversarial_loss(real_guess, torch.ones_like(real_guess))

        fake_guess = self.discriminator(poisoned_images.detach())
        loss_D_fake = self.adversarial_loss(fake_guess, torch.zeros_like(fake_guess))

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()

        # --- Generator step ---
        self.optimizer_G.zero_grad()

        fake_guess_again = self.discriminator(poisoned_images)
        loss_G_fake = self.adversarial_loss(fake_guess_again, torch.ones_like(fake_guess_again))

        victim_predictions = self.target_model(poisoned_images)
        loss_adv = -F.cross_entropy(victim_predictions, true_labels)

        # 0.5 weight keeps the victim loss from overpowering the stealth loss.
        loss_G = loss_G_fake + (0.5 * loss_adv)
        loss_G.backward()
        self.optimizer_G.step()

        return loss_G.item(), loss_D.item()