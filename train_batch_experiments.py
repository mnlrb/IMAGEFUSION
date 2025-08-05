import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
import os
import time
import psutil


EXPERIMENTS = [
    ("rgb_only_resnet18", "single", "resnet18"),
    ("ir_only_resnet18", "single", "resnet18"),
    ("rgb_only_mobilenet", "single", "mobilenet"),
    ("ir_only_mobilenet", "single", "mobilenet"),
    ("fusion_early_resnet18", "early", "resnet18"),
    ("fusion_late_resnet18", "late", "resnet18"),
    ("fusion_late_mobilenet", "late", "mobilenet"),
    ("fusion_late_vgg16", "late", "vgg16"),
    ("fusion_late_efficientnet", "late", "efficientnet"),
]

SAMPLES_PER_CLASS = 2000
EPOCHS = 30
BATCH_SIZE = 64
DATA_DIR = "D:/frames"
LABELS_CSV = "D:/frame_labels_aligned.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset definition
class FlameDataset(Dataset):
    def __init__(self, frame_dir, labels_csv, mode="train", fusion="late", transform=None, use_rgb=True, use_ir=True):
        self.frame_dir = Path(frame_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.fusion = fusion
        self.use_rgb, self.use_ir = use_rgb, use_ir

        self.label_map = {"NN": 0, "YN": 1, "YY": 2}
        self.labels_df = self.labels_df[self.labels_df.label.isin(self.label_map)]
        self.labels_df["label_id"] = self.labels_df["label"].map(self.label_map)

        max_frame_id = min(
            len(list((self.frame_dir / "RGB").glob("*.jpg"))),
            len(list((self.frame_dir / "IR").glob("*.jpg")))
        )
        self.labels_df = self.labels_df[self.labels_df["frame_id"] <= max_frame_id]

        self.labels_df = (
            self.labels_df.groupby("label_id", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=42))
            .reset_index(drop=True)
        )

        train_idx, val_idx = train_test_split(
            np.arange(len(self.labels_df)),
            test_size=0.2,
            stratify=self.labels_df["label_id"],
            random_state=42
        )

        self.indices = train_idx if mode == "train" else val_idx
        print(f"[{mode}] samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[self.indices[idx]]
        fname = f"{row.frame_id:06d}.jpg"

        imgs = []
        if self.use_rgb:
            rgb = Image.open(self.frame_dir / "RGB" / fname).convert("RGB")
            if self.transform:
                rgb = self.transform(rgb)
            imgs.append(rgb)
        if self.use_ir:
            ir = Image.open(self.frame_dir / "IR" / fname).convert("RGB")
            if self.transform:
                ir = self.transform(ir)
            imgs.append(ir)

        if self.fusion == "early" and len(imgs) == 2:
            x = torch.cat(imgs, dim=0)  # merge to 6 channels
        elif len(imgs) == 1:
            x = imgs[0]
        else:
            x = tuple(imgs)

        return x, row.label_id

# Model for late fusion
class LateFusionNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=3):
        super().__init__()

        def _load_backbone(name):
            if name == "mobilenet":
                net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                feat_dim = net.classifier[1].in_features
                net.classifier = nn.Identity()
            elif name == "vgg16":
                net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                feat_dim = net.classifier[6].in_features
                net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])
            elif name == "efficientnet":
                net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                feat_dim = net.classifier[1].in_features
                net.classifier = nn.Identity()
            else:
                net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                feat_dim = net.fc.in_features
                net.fc = nn.Identity()
            return net, feat_dim

        self.rgb_net, feat_dim = _load_backbone(backbone)
        self.ir_net, _ = _load_backbone(backbone)
        self.fc = nn.Linear(feat_dim * 2, num_classes)

    def forward(self, x):
        rgb, ir = x
        feat1 = self.rgb_net(rgb)
        feat2 = self.ir_net(ir)
        feat = torch.cat([feat1, feat2], dim=1)
        return self.fc(feat)

# Model for single modality and early fusion
class SingleModalityNet(nn.Module):
    def __init__(self, backbone="resnet18", fusion="single", num_classes=3):
        super().__init__()
        if backbone == "mobilenet":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif backbone == "vgg16":
            net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            feat_dim = net.classifier[6].in_features
            net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])
        elif backbone == "efficientnet":
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            if fusion == "early":
                old_conv = net.conv1
                net.conv1 = nn.Conv2d(6, old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias)
                with torch.no_grad():
                    net.conv1.weight[:, :3] = old_conv.weight
                    net.conv1.weight[:, 3:] = old_conv.weight

        self.backbone = net
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)

# Save confusion matrix
def save_confusion_matrix(y_true, y_pred, exp_name, classes=["NN", "YN", "YY"]):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {exp_name}")
    cm_dir = os.path.join(OUTPUT_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    filename = os.path.join(cm_dir, f"{exp_name}_confusion_matrix.png")
    plt.savefig(filename)
    plt.close()
    return filename

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for exp_name, fusion, backbone in EXPERIMENTS:
        print(f"\n--- Starting experiment: {exp_name} ---")
        start_time_total = time.time()

        use_rgb, use_ir = True, True
        if "rgb_only" in exp_name:
            use_rgb, use_ir = True, False
        elif "ir_only" in exp_name:
            use_rgb, use_ir = False, True

        train_dataset = FlameDataset(DATA_DIR, LABELS_CSV, mode="train", fusion=fusion,
                                     transform=transform, use_rgb=use_rgb, use_ir=use_ir)
        val_dataset = FlameDataset(DATA_DIR, LABELS_CSV, mode="val", fusion=fusion,
                                   transform=transform, use_rgb=use_rgb, use_ir=use_ir)

        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        }

        if fusion == "late":
            model = LateFusionNet(backbone=backbone, num_classes=3).to(DEVICE)
        else:
            model = SingleModalityNet(backbone=backbone, fusion=fusion, num_classes=3).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        log_file = os.path.join(OUTPUT_DIR, f"training_log_{exp_name}.csv")
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Phase", "Loss", "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrixFile", "EpochTime(s)", "MemoryUsage(MB)"])

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epoch_start = time.time()
            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()
                running_loss, running_corrects = 0, 0
                all_preds, all_labels = [], []

                for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                    labels = labels.to(DEVICE)
                    if fusion == "late":
                        inputs = (inputs[0].to(DEVICE), inputs[1].to(DEVICE))
                    else:
                        inputs = inputs.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_time = time.time() - epoch_start
                mem_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

                if phase == "train":
                    print(f"{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f} Time {epoch_time:.1f}s Mem {mem_usage:.1f}MB")
                    prec, rec, f1, cm_file = np.nan, np.nan, np.nan, ""
                else:
                    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
                    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
                    cm_file = save_confusion_matrix(all_labels, all_preds, exp_name)
                    print(f"{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f} Prec {prec:.4f} Rec {rec:.4f} F1 {f1:.4f} Time {epoch_time:.1f}s Mem {mem_usage:.1f}MB")

                with open(log_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item(), prec, rec, f1, cm_file, round(epoch_time, 2), round(mem_usage, 1)])

        total_time = time.time() - start_time_total
        print(f"Total training time for {exp_name}: {total_time:.1f}s")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{exp_name}_model.pth"))
