import torch
import torch.nn as nn
import os
import numpy as np
import pygame
from PIL import Image, ImageFilter

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition eines CNN-Modells für die Bildvorhersage
class PixelPredictionCNN(nn.Module):
    def __init__(self):
        super(PixelPredictionCNN, self).__init__()
        # Encoder: Extrahiert Merkmale aus dem Eingabebild
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bottleneck: weitere Merkmalverdichtung
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder: Rekonstruiert das Bild aus den Merkmalen
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=9, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x

current_dir = os.path.dirname(os.path.realpath(__file__))
# Absoluter Pfad zum Modell
MODEL_PATH = os.path.join(current_dir, 'model', 'plasma_model.pth')

def load_model(model, filename=MODEL_PATH):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        print(f"✅ Modell {filename} geladen")
        return True
    else:
        print("⚠️ Kein gespeichertes Modell gefunden. Es wird ein zufälliges Modell verwendet.")
        return False

# Initialisiere das Modell und lade ggf. ein vortrainiertes Modell
model = PixelPredictionCNN().to(device)
load_model(model)

# Noise-Parameter für den Ornstein-Uhlenbeck Prozess
Noize = 0.325

# Klasse für stabilen, zeitabhängigen Noise
class StableNoise:
    def __init__(self, shape, theta=0.0211, sigma=Noize, mu=0.1, dt=1.1):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.noise = torch.zeros(shape).to(device)

    def reset(self):
        self.noise = torch.zeros_like(self.noise)

    def sample(self):
        normal_sample = torch.randn_like(self.noise).to(device)
        self.noise += self.theta * (self.mu - self.noise) * self.dt + self.sigma * (self.dt ** 0.5) * normal_sample
        return self.noise
    def update(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
stable_noise = None
theta_run = 0.015
Noize_run = 0.375
# Funktion zur Vorhersage des nächsten Bildes anhand des aktuellen Eingabebildes
def predict_next_image(input_image):
    global stable_noise
    global theta_run
    global Noize_run
    input_image = input_image.to(device)
    latent = model.encoder(input_image)
    
    # Initialisiere Noise-Objekt, falls noch nicht vorhanden oder bei Formänderung
    if stable_noise is None or stable_noise.noise.shape != latent.shape:
        stable_noise = StableNoise(latent.shape)
    theta_run += 0.001
    if theta_run > 0.06:
        theta_run = 0.009
    Noize_run += 0.001
    if Noize_run > 0.6:
        Noize_run = 0.475
    stable_noise.update(theta_run, Noize)    
    noise = stable_noise.sample()
    latent_noisy = latent + noise
    with torch.no_grad():
        output = model.decoder(latent_noisy)
    # Für die weitere Animation wird Gradientenberechnung nicht benötigt
    return output

# Funktion, um ein Bild-Tensor in einem Pygame-Fenster anzuzeigen
def display_image(image_tensor):
    """
    Konvertiert das Bild aus dem GPU-Tensor in ein Pygame-kompatibles Format.
    """
    image_array = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    
    scale_factor = 16
    image_array = np.repeat(np.repeat(image_array, scale_factor, axis=0), scale_factor, axis=1)
    
    surface = pygame.surfarray.make_surface(image_array)
    # Optional: Gaußscher Unschärfefilter
    surface_array = pygame.surfarray.array3d(surface)
    surface_array = np.transpose(surface_array, (1, 0, 2))
    pil_img = Image.fromarray(surface_array)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=8))
    new_array = np.array(pil_img)
    new_array = np.transpose(new_array, (1, 0, 2))
    surface = pygame.surfarray.make_surface(new_array)
    
    screen.blit(surface, (0, 0))
    pygame.display.flip()

# Pygame-Initialisierung und Fensterkonfiguration
pygame.init()
window_size = (640, 640)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Realtime PlasmaClouds AI")
clock = pygame.time.Clock()
FPS = 60

# Initialisiere ein Startbild als Zufallsrauschen
old_image = torch.rand(1, 3, 40, 40).to(device)

# Hauptschleife zur Anzeige der Animation
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Erzeuge das nächste Bild basierend auf dem aktuellen
    predicted_image = predict_next_image(old_image)
    old_image = predicted_image.detach()
    
    display_image(predicted_image)
    clock.tick(FPS)

print("Programm beendet!")
pygame.quit()
