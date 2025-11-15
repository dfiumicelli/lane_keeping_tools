"""
Training U-Net 3-CLASSI - ILLUMINATION ROBUST
Data Augmentation su illuminazione
Input: RGB + Sobel Edges per shape robustness
Output: Rete resiliente a qualsiasi condizione di illuminazione
"""

import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== LOSS FUNCTIONS ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else torch.ones(3))
        self.ce = None

    def forward(self, inputs, targets):
        device = inputs.device

        if self.ce is None or self.alpha.device != device:
            alpha_device = self.alpha.to(device)
            self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha_device)

        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.6, dice_weight=0.4, num_classes=3):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes

        class_weights = torch.tensor([1.0, 3.0, 5.0])
        self.register_buffer('class_weights', class_weights)
        self.focal = None

    def forward(self, predictions, targets):
        device = predictions.device

        if self.focal is None:
            class_weights_device = self.class_weights.to(device)
            self.focal = FocalLoss(alpha=class_weights_device, gamma=2.5)

        if self.focal.alpha.device != device:
            class_weights_device = self.class_weights.to(device)
            self.focal = FocalLoss(alpha=class_weights_device, gamma=2.5)

        focal = self.focal(predictions, targets)

        probs = torch.softmax(predictions, dim=1)
        dice_loss = 0

        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()

            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + 1.0) / (pred_c.sum() + target_c.sum() + 1.0)
            dice_loss += 1 - dice

        dice_loss /= self.num_classes

        return self.focal_weight * focal + self.dice_weight * dice_loss


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.defaults['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ==================== DATASET CON ILLUMINATION AUGMENTATION ====================

class LaneDataset3ClassIlluminationRobust(Dataset):
    """Dataset SHAPE-BASED con augmentation PESANTE su illuminazione"""

    def __init__(self, images_dir, masks_dir, img_size=(320, 240), is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.is_train = is_train

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        self.mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        self.num_pairs = min(len(self.image_files), len(self.mask_files))

    def __len__(self):
        return self.num_pairs

    def color_to_class(self, mask_rgb):
        """RGB → classe indice"""
        mask_class = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

        white = (mask_rgb[:, :, 0] == 255) & (mask_rgb[:, :, 1] == 255) & (mask_rgb[:, :, 2] == 255)
        mask_class[white] = 1

        yellow = (mask_rgb[:, :, 0] == 0) & (mask_rgb[:, :, 1] == 255) & (mask_rgb[:, :, 2] == 255)
        mask_class[yellow] = 2

        return mask_class

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(mask_path)
        if mask_rgb is None:
            return self.__getitem__(0)

        # Ridimensiona
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
        mask_rgb = cv2.resize(mask_rgb, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)

        mask_class = self.color_to_class(mask_rgb)
        mask_uint8 = mask_class.astype(np.uint8)

        if self.is_train:
            #AUGMENTATION PESANTE SU ILLUMINAZIONE - VERSIONE CORRETTA 
            transform = A.Compose([
                # ===== GEOMETRIC AUGMENTATION =====
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=8, p=0.2),

                # ===== ILLUMINATION AUGMENTATION=====

                # 1.  Brightness/Contrast variations
                A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.9),

                # 2.  Gamma correction (simula diverse telecamere)
                A.RandomGamma(gamma_limit=(40, 200), p=0.8),

                # 3.  Equalizzazione istogrammi (VERIFICATO)
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),

                # 4.  Blur variations 
                A.MotionBlur(blur_limit=7, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.2),

                # 5.  Shadows/Highlights
                A.RandomShadow(p=0.4),

                # 6.  Downscale + upscale 
                A.Downscale(scale_range=(0.7, 0.9), p=0.3),

                # 7.  Channel shifts (variabilità sensore)
                A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.7),

                # 8.  Hue/Saturation variations
                A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=0.8),

                # 9.  Image compression artifacts 
                A.ImageCompression(quality_range=(30, 90), p=0.4),

                # 10.  Noise 
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(multiplier=(0.5, 2.0), p=0.3),

                # 11.  Posterize (Riduzione bit depth) 
                A.Posterize(num_bits=4, p=0.2),

                # 12.  Solarize (Invert intensity) 
                A.Solarize(threshold_range=(0.0, 1.0), p=0.15),

                # ===== FINAL NORMALIZATION =====
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            # Validation: solo normalization (no augmentation)
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})

        # Applica augmentation
        sample = transform(image=image, mask=mask_uint8)
        image_rgb = sample['image']
        mask_tensor = sample['mask'].long()

        #  AGGIUNGI CANALE SOBEL EDGES
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges = (edges / edges.max()).astype(np.float32) if edges.max() > 0 else edges.astype(np.float32)

        edges_tensor = torch.from_numpy(edges).unsqueeze(0)

        #  CONCATENA: RGB + EDGES = 4 CANALI
        image_with_edges = torch.cat([image_rgb, edges_tensor], dim=0)

        return image_with_edges, mask_tensor


# ==================== MODELLO CUSTOM ====================

class UNetWith4Channels(nn.Module):
    """U-Net che prende 4 canali in input (RGB + Edges)"""

    def __init__(self, encoder_name='mobilenet_v2', num_classes=3, dropout=0.3):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_dropout=dropout,
        )

        self.conv_4to3 = nn.Conv2d(4, 3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv_4to3(x)
        x = self.unet(x)
        return x


# ==================== CONFIGURAZIONE ====================

class Config:
    DATASET_DIR = '/kaggle/input/turtebot-dataset/turtlebot_dataset' #aggiungere path dataset
    IMAGES_DIR = os.path.join(DATASET_DIR, 'test_images')
    MASKS_DIR = os.path.join(DATASET_DIR, 'mask_images_auto')

    TRAIN_RATIO = 0.9
    IMG_SIZE = (320, 240)

    ENCODER = 'mobilenet_v2'
    CLASSES = 3

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    EPOCHS = 80
    BATCH_SIZE = 24
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.35
    NUM_WORKERS = 2

    USE_AMP = True
    WARMUP_EPOCHS = 15

    PATIENCE = 30
    MIN_DELTA = 0.0001

    MODEL_DIR = '/kaggle/working' #path output
    BEST_MODEL_PTH = os.path.join(MODEL_DIR, 'best_unet_3class_illumination_robust.pth')
    BEST_MODEL_ONNX = os.path.join(MODEL_DIR, 'best_unet_3class_illumination_robust.onnx')
    LAST_MODEL_PTH = os.path.join(MODEL_DIR, 'last_unet_3class_illumination_robust.pth')
    LAST_MODEL_ONNX = os.path.join(MODEL_DIR, 'last_unet_3class_illumination_robust.onnx')


# ==================== TRAINING ====================

def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        if use_amp and scaler:
            with autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_fn, device, num_classes=3):
    model.eval()
    total_loss = 0
    iou_per_class = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            for c in range(num_classes):
                pred_c = (preds == c).float().flatten()
                mask_c = (masks == c).float().flatten()

                intersection = (pred_c * mask_c).sum()
                union = pred_c.sum() + mask_c.sum() - intersection
                iou = intersection / (union + 1e-6)
                iou_per_class[c].append(iou.item())

    avg_loss = total_loss / len(val_loader)
    avg_iou_per_class = [np.mean(iou_per_class[c]) if iou_per_class[c] else 0.0 for c in range(num_classes)]
    mean_iou = np.mean(avg_iou_per_class)

    return avg_loss, mean_iou, avg_iou_per_class


def export_onnx(model, onnx_path, device, img_size=(320, 240)):
    model.eval()
    dummy_input = torch.randn(1, 4, img_size[1], img_size[0], device=device)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False
    )
    print(f"   ONNX: {os.path.basename(onnx_path)}")


# ==================== MAIN ====================

def main():
    config = Config()

    print("\n" + "=" * 80)
    print("🚀 TRAINING U-NET 3-CLASSI - ILLUMINATION ROBUST (VERSIONE CORRETTA)")
    print("=" * 80 + "\n")

    if not os.path.exists(config.IMAGES_DIR) or not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"Dataset not found!")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print(" Caricamento dataset...\n")
    full_dataset = LaneDataset3ClassIlluminationRobust(config.IMAGES_DIR, config.MASKS_DIR, config.IMG_SIZE,
                                                       is_train=True)

    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size

    print(f" Dataset: {total_size} coppie")
    print(f" Split: {train_size} train / {val_size} val")
    print(f" Input: 4 canali (RGB + Sobel Edges)")
    print(f" AUGMENTATION (12 strategie verificate):")
    print(f"   1. Brightness/Contrast: ±60%")
    print(f"   2. Gamma corrections: 40-200")
    print(f"   3. CLAHE: Equalizzazione adaptiva")
    print(f"   4. Motion/Median Blur: Movimento e rumore")
    print(f"   5. Random Shadows: Ombre artificiali")
    print(f"   6. Downscale: Compressione simulata")
    print(f"   7. RGB Shifts: ±40 per canale")
    print(f"   8. Hue/Saturation: ±50")
    print(f"   9. JPEG Compression: Quality 30-95")
    print(f"   10. Noise: Gaussiano + Moltiplicativo")
    print(f"   11. Posterize: Riduzione bit depth")
    print(f"   12. Solarize: Inversione intensità\n")

    train_dataset_final = LaneDataset3ClassIlluminationRobust(config.IMAGES_DIR, config.MASKS_DIR, config.IMG_SIZE,
                                                              is_train=True)
    val_dataset_final = LaneDataset3ClassIlluminationRobust(config.IMAGES_DIR, config.MASKS_DIR, config.IMG_SIZE,
                                                            is_train=False)

    train_subset, _ = random_split(train_dataset_final, [train_size, val_size],
                                   generator=torch.Generator().manual_seed(42))
    _, val_subset = random_split(val_dataset_final, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    print(" Creazione modello (4 canali: RGB + Edges)...\n")
    model = UNetWith4Channels(
        encoder_name=config.ENCODER,
        num_classes=config.CLASSES,
        dropout=config.DROPOUT,
    )

    model.to(config.DEVICE)

    loss_fn = CombinedLoss(focal_weight=0.6, dice_weight=0.4, num_classes=3)
    loss_fn.to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, config.WARMUP_EPOCHS, config.EPOCHS, min_lr=5e-8)
    scaler = GradScaler() if config.USE_AMP else None

    best_mean_iou = 0.0
    no_improve_count = 0

    print(f" CONFIGURAZIONE:")
    print(f"   Strategy: ILLUMINATION ROBUST")
    print(f"   Input: 4 canali (RGB + Sobel Edges)")
    print(f"   LR: {config.LEARNING_RATE:.6f}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Dropout: {config.DROPOUT}\n")
    print("  Inizio training...\n")

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("=" * 70)

        current_lr = scheduler.step(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE, scaler, config.USE_AMP)
        val_loss, mean_iou, iou_per_class = validate_epoch(model, val_loader, loss_fn, config.DEVICE, num_classes=3)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Mean IoU: {mean_iou:.4f} | BG: {iou_per_class[0]:.4f} | Continua: {iou_per_class[1]:.4f} | Discontinua: {iou_per_class[2]:.4f}")
        print(f"LR: {current_lr:.6f}")

        torch.save(model.state_dict(), config.LAST_MODEL_PTH)
        export_onnx(model, config.LAST_MODEL_ONNX, config.DEVICE, config.IMG_SIZE)

        if mean_iou > best_mean_iou + config.MIN_DELTA:
            best_mean_iou = mean_iou
            no_improve_count = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PTH)
            export_onnx(model, config.BEST_MODEL_ONNX, config.DEVICE, config.IMG_SIZE)
            print(f" NEW BEST! IoU: {best_mean_iou:.4f} | Discontinua: {iou_per_class[2]:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement ({no_improve_count}/{config.PATIENCE})")

            if no_improve_count >= config.PATIENCE:
                print(f"\nEARLY STOPPING!")
                break

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETATO!")
    print("=" * 80)
    print(f" Best Mean IoU: {best_mean_iou:.4f}\n")
    print(f" Modelli salvati in: {config.MODEL_DIR}\n")
    print(f"   best_unet_3class_illumination_robust.pth")
    print(f"   best_unet_3class_illumination_robust.onnx")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
