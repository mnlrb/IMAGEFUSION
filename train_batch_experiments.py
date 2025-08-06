import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import csv
import yaml

# Load config
with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_CSV = config["data"]["csv_file"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
IMG_SIZE = config["training"]["image_size"]
LR = config["training"]["learning_rate"]
OUTPUT_DIR = config["logging"]["output_dir"]
EXPERIMENTS = config["experiments"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary_results.csv")

with open(SUMMARY_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Val_Loss", "Precision", "Recall", "F1"])

# Dataset class
class FlameDataset(Dataset):
    def __init__(self, csv_file, mode="train", transform=None, use_rgb=True, use_ir=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_rgb, self.use_ir = use_rgb, use_ir
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(0.8 * len(self.data))
        self.data = self.data.iloc[:n_train] if mode == "train" else self.data.iloc[n_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        imgs = []
        if self.use_rgb:
            rgb = Image.open(row.rgb_path).convert("RGB")
            if self.transform: rgb = self.transform(rgb)
            imgs.append(rgb)
        if self.use_ir:
            ir = Image.open(row.ir_path).convert("RGB")
            if self.transform: ir = self.transform(ir)
            imgs.append(ir)
        if self.use_rgb and self.use_ir and len(imgs) == 2:
            x = torch.cat(imgs, dim=0)
        else:
            x = imgs[0]
        y = torch.tensor([row.fire, row.smoke], dtype=torch.float32)
        return x, y

# Early fusion model
class EarlyFusionNet(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features
        self.fc = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.fc(feat)

# Late fusion model
class LateFusionNet(nn.Module):
    def __init__(self, backbone="resnet18"):
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
        self.fc = nn.Linear(feat_dim * 2, 2)

    def forward(self, x):
        rgb, ir = x
        feat1 = self.rgb_net(rgb)
        feat2 = self.ir_net(ir)
        feat = torch.cat([feat1, feat2], dim=1)
        return self.fc(feat)

# Single modality model
class SingleModalityNet(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        if backbone == "mobilenet":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        self.backbone = net
        self.fc = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)

# Confusion matrix saving
def save_confusion_matrices(y_true, y_pred, exp_name):
    cms = multilabel_confusion_matrix(y_true, y_pred)
    classes = ["Fire", "Smoke"]
    cm_dir = os.path.join(OUTPUT_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    for idx, cm in enumerate(cms):
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No","Yes"], yticklabels=["No","Yes"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{exp_name} - {classes[idx]}")
        plt.savefig(os.path.join(cm_dir, f"{exp_name}_{classes[idx]}_cm.png"))
        plt.close()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    summary_data = []

    for exp in EXPERIMENTS:
        exp_name, fusion, backbone = exp["name"], exp["fusion"], exp["backbone"]
        print(f"\n--- Starting experiment: {exp_name} ---")

        use_rgb, use_ir = True, True
        if "rgb_only" in exp_name: use_ir = False
        if "ir_only" in exp_name: use_rgb = False

        train_dataset = FlameDataset(DATA_CSV, mode="train", transform=transform, use_rgb=use_rgb, use_ir=use_ir)
        val_dataset = FlameDataset(DATA_CSV, mode="val", transform=transform, use_rgb=use_rgb, use_ir=use_ir)

        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        }

        if fusion == "early":
            model = EarlyFusionNet(backbone=backbone).to(DEVICE)
        elif fusion == "late":
            model = LateFusionNet(backbone=backbone).to(DEVICE)
        else:
            model = SingleModalityNet(backbone=backbone).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        log_file = os.path.join(OUTPUT_DIR, f"{exp_name}_training_log.csv")
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "f1"])

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            metrics = {}
            for phase in ["train", "val"]:
                model.train() if phase=="train" else model.eval()
                running_loss = 0
                all_preds, all_labels = [], []

                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    labels = labels.to(DEVICE)
                    if fusion=="late" and use_rgb and use_ir:
                        inputs = (inputs[:, :3, :, :].to(DEVICE), inputs[:, 3:, :, :].to(DEVICE))
                    else:
                        inputs = inputs.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=="train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = (torch.sigmoid(outputs) > 0.5).int()
                        if phase=="train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                metrics[f"{phase}_loss"] = epoch_loss

                if phase=="val":
                    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
                    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
                    metrics.update({"precision": prec, "recall": rec, "f1": f1})

                    save_confusion_matrices(np.array(all_labels), np.array(all_preds), exp_name)

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, metrics["train_loss"], metrics["val_loss"],
                                 metrics["precision"], metrics["recall"], metrics["f1"]])

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{exp_name}_model.pth"))
        summary_data.append([exp_name, metrics["val_loss"], metrics["precision"], metrics["recall"], metrics["f1"]])
        print(f"{exp_name} training complete.")

    summary_df = pd.DataFrame(summary_data, columns=["Experiment", "Val_Loss", "Precision", "Recall", "F1"])
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"Summary results saved to {SUMMARY_FILE}")
