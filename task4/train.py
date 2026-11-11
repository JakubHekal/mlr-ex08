"""Task 4: what does each change bring to the Task 3 leaf classifier?

Baselines come first, then one change at a time, then combinations. Every run
trains on the same images with a fixed seed and scores the same four fixed
sets, so the rows in results.csv are comparable:

  valA      the 28 course validation images
  cameraA   each valA image cropped, rotated and recoloured once (5 copies)
  testB     another group's photos, never trained on in A-only rows
  testH     last years' photos, never trained on in A-only rows

Usage:
  python task4/train.py --prepare
  python task4/train.py --run B1 --seed 0
  python task4/train.py --summary
  python task4/train.py --export L_combo --seed 0

Task 3 code (LeafDataset, LeafNetBN, training loop) is copied from
CC8/ex08_student.ipynb cells 55, 59, 61 and 62. Only the switches a variant
needs are new.
"""

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # torch.onnx prints emoji on Windows

ROOT = Path(__file__).resolve().parents[1]       # ex08_for_agent
DATA = ROOT.parent / "data"                      # CC8/data
A_DIR = DATA / "leaves"
B_DIR = DATA / "external" / "group_b" / "leaves"
H_DIR = DATA / "external" / "historical" / "leaves"
CAMERA_DIR = DATA / "camera_val"
RESULTS = ROOT / "task4" / "results.csv"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SETS = {"A": A_DIR, "B": B_DIR, "H": H_DIR}
ZIPS = {
    "group_b": "https://drive.usercontent.google.com/download?id=1UDLwa3gJL9nWEiZ-6WbrbMk-yQvExgv_&export=download&confirm=t",
    "historical": "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip",
}

plain_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
geo_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
color_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
AUGS = {"none": plain_transform, "geo": geo_transform, "color": color_transform}

# Family "leaf" starts from the Task 3 network, family "mob" from the Task 2
# model. Within a family every variant differs from its parent in one field.
VARIANTS = {
    # baselines
    "B1": dict(family="leaf", parent="-", change="baseline", net="leaf", aug="geo", epochs=15),
    "B2": dict(family="mob", parent="-", change="baseline", net="mob", aug="none", epochs=15),
    # family 1: one change each, against B1
    "L_color": dict(family="leaf", parent="B1", change="colour jitter", aug="color"),
    "L_drop": dict(family="leaf", parent="B1", change="dropout 0.3", dropout=0.3),
    "L_gap": dict(family="leaf", parent="B1", change="global average pooling", gap=True),
    "L_adamw": dict(family="leaf", parent="B1", change="AdamW, weight decay 1e-2", opt="adamw"),
    "L_ls": dict(family="leaf", parent="B1", change="label smoothing 0.1", label_smoothing=0.1),
    "L_best": dict(family="leaf", parent="B1", change="best epoch weights", best_epoch=True),
    "L_cos": dict(family="leaf", parent="B1", change="cosine schedule, 30 epochs", cosine=True, epochs=30),
    # family 2: one change each, against B2
    "M_frozen": dict(family="mob", parent="B2", change="frozen backbone", frozen=True),
    "M_aug": dict(family="mob", parent="B2", change="geometric augmentation", aug="geo"),
    "M_cos": dict(family="mob", parent="B2", change="cosine schedule, 30 epochs", cosine=True, epochs=30),
    "M_best": dict(family="mob", parent="B2", change="best epoch weights", best_epoch=True),
    # extra training images, counted separately from the one-change rows
    "L_dataB": dict(family="leaf", parent="B1", change="training images +B", train_set="AB"),
    "L_dataH": dict(family="leaf", parent="B1", change="training images +H", train_set="AH"),
    "L_dataBH": dict(family="leaf", parent="B1", change="training images +B+H", train_set="ABH"),
    "M_dataB": dict(family="mob", parent="B2", change="training images +B", train_set="AB"),
    "M_dataH": dict(family="mob", parent="B2", change="training images +H", train_set="AH"),
    "M_dataBH": dict(family="mob", parent="B2", change="training images +B+H", train_set="ABH"),
    # combinations, filled in after the single-change rows are measured
    "L_combo1": dict(family="leaf", parent="B1", change="colour jitter + best epoch", aug="color", best_epoch=True),
    "L_combo2": dict(family="leaf", parent="B1", change="colour jitter + label smoothing + best epoch",
                     aug="color", label_smoothing=0.1, best_epoch=True),
    "M_combo": dict(family="mob", parent="B2", change="augmentation + cosine 30 + best epoch",
                    aug="geo", cosine=True, epochs=30, best_epoch=True),
}
# Fill defaults once, so each variant entry names only what it changes.
for cfg in VARIANTS.values():
    cfg.setdefault("net", "leaf" if cfg["family"] == "leaf" else "mob")
    cfg.setdefault("aug", "geo")
    cfg.setdefault("epochs", 15)
    cfg.setdefault("train_set", "A")
    cfg.setdefault("opt", "adam")


class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted(path.name for path in Path(folder).iterdir() if path.is_dir())
        self.samples = []
        extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
        for class_index, class_name in enumerate(self.classes):
            class_folder = Path(folder) / class_name
            for image_path in sorted(class_folder.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in extensions:
                    self.samples.append((str(image_path), class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.transform(Image.open(self.samples[i][0]).convert("RGB")), self.samples[i][1]


class LeafNetBN(nn.Module):
    def __init__(self, gap=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = gap
        self.fc1 = nn.Linear(64 if gap else 64 * 6 * 6, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.mean(dim=(2, 3)) if self.gap else x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def build_model(cfg):
    if cfg["net"] == "leaf":
        model = LeafNetBN(gap=cfg.get("gap", False), dropout=cfg.get("dropout", 0.0))
    else:
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        if cfg.get("frozen", False):
            for parameter in model.features.parameters():
                parameter.requires_grad = False
    return model


def build_optimizer(model, cfg):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if cfg["opt"] == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
    scheduler = None
    if cfg.get("cosine", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    return optimizer, scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loader(dataset, shuffle, seed=0, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def accuracy(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(data_loader.dataset)


def validation_loss(model, data_loader):
    model.eval()
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in data_loader:
            loss_sum += criterion(model(images), labels).item() * len(labels)
    return loss_sum / len(data_loader.dataset)


def train_variant(cfg, seed):
    set_seed(seed)
    model = build_model(cfg)
    optimizer, scheduler = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    train_ds = ConcatDataset([LeafDataset(SETS[name] / "train", AUGS[cfg["aug"]])
                              for name in cfg["train_set"]])
    train_loader = loader(train_ds, shuffle=True, seed=seed)
    valA_loader = loader(LeafDataset(A_DIR / "val", plain_transform), shuffle=False)

    best_loss, best_state = float("inf"), None
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if cfg.get("best_epoch", False):
            loss = validation_loss(model, valA_loader)
            if loss < best_loss:
                best_loss, best_state = loss, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def scoring_loaders():
    if not CAMERA_DIR.exists() or len(list(CAMERA_DIR.rglob("*.png"))) == 0:
        raise SystemExit("camera_val missing. Run --prepare first.")
    return {
        "valA": loader(LeafDataset(A_DIR / "val", plain_transform), shuffle=False),
        "cameraA": loader(LeafDataset(CAMERA_DIR, plain_transform), shuffle=False),
        "testB": loader(LeafDataset(B_DIR / "val", plain_transform), shuffle=False),
        "testH": loader(LeafDataset(H_DIR / "val", plain_transform), shuffle=False),
    }


def write_row(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS.exists()
    columns = ["family", "variant", "parent", "change", "train_set", "epochs",
               "seed", "valA", "cameraA", "testB", "testH", "seconds"]
    with open(RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(name, seed):
    cfg = VARIANTS[name]
    print(f"{name} seed {seed}: {cfg['change']} on train set {cfg['train_set']}")
    start = time.perf_counter()
    model = train_variant(cfg, seed)
    loaders = scoring_loaders()
    scores = {key: accuracy(model, value) for key, value in loaders.items()}
    seconds = time.perf_counter() - start
    row = dict(family=cfg["family"], variant=name, parent=cfg["parent"], change=cfg["change"],
               train_set=cfg["train_set"], epochs=cfg["epochs"], seed=seed,
               seconds=f"{seconds:.1f}", **{k: f"{v:.4f}" for k, v in scores.items()})
    # A test set that was in the training data is no test.
    if "B" in cfg["train_set"]:
        row["testB"] = "in-train"
    if "H" in cfg["train_set"]:
        row["testH"] = "in-train"
    write_row(row)
    print("  " + "  ".join(f"{key} {value}" for key, value in scores.items()) + f"  {seconds:.1f} s")
    return model


def read_rows():
    with open(RESULTS, newline="") as f:
        return list(csv.DictReader(f))


def summary():
    rows = read_rows()
    repeats = {}
    for row in rows:
        key = (row["variant"], row["train_set"])
        repeats.setdefault(key, []).append(row)

    means = {}
    for (variant, train_set), runs in repeats.items():
        means[(variant, train_set)] = {
            "parent": runs[0]["parent"], "change": runs[0]["change"], "runs": len(runs),
            **{metric: (np.mean([float(r[metric]) for r in runs if r[metric] not in ("", "in-train")]),
                       np.std([float(r[metric]) for r in runs if r[metric] not in ("", "in-train")]))
               for metric in ("valA", "cameraA", "testB", "testH")
               if any(r[metric] not in ("", "in-train") for r in runs)}
        }

    print("\nRows repeated over seeds (mean +- std):")
    print("| variant | train | change | seeds | valA | cameraA | testB | testH |")
    print("|---|---|---|---|---|---|---|---|")
    for (variant, train_set), m in means.items():
        if m["runs"] < 2:
            continue
        cells = [f"{m[k][0]:.3f} +- {m[k][1]:.3f}" if k in m else "-"
                 for k in ("valA", "cameraA", "testB", "testH")]
        print(f"| {variant} | {train_set} | {m['change']} | {m['runs']} | " + " | ".join(cells) + " |")

    print("\nSingle runs and their change against the parent mean:")
    print("| variant | train | change | seed | valA | cameraA | testB | testH |")
    print("|---|---|---|---|---|---|---|---|")
    for (variant, train_set), runs in repeats.items():
        for row in runs:
            cells = [row[k] for k in ("valA", "cameraA", "testB", "testH")]
            delta = ""
            parent = means.get((row["parent"], train_set))
            if parent and "valA" in parent and row["valA"] not in ("", "in-train"):
                delta = f" ({float(row['valA']) - parent['valA'][0]:+.3f} valA)"
            print(f"| {variant} | {train_set} | {row['change']}{delta} | {row['seed']} | "
                  + " | ".join(cells) + " |")


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in ZIPS.items():
        target = DATA / "external" / name
        if (target / "leaves").exists():
            print(f"{name}: already extracted")
            continue
        print(f"{name}: downloading")
        archive = DATA / "external" / f"{name}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink()
        print(f"{name}: extracted to {target}")

    for name, root in SETS.items():
        for split in ("train", "val"):
            dataset = LeafDataset(root / split, plain_transform)
            print(f"{name} {split}: {len(dataset)} images, classes {dataset.classes}")

    make_camera_val()
    counts = {path.name: len(list(path.iterdir())) for path in CAMERA_DIR.iterdir() if path.is_dir()}
    print(f"cameraA: {sum(counts.values())} images {counts}")


def make_camera_val(copies=5):
    """One fixed camera-like copy set: 5 per valA image, created with seed 0."""
    expected = copies * len(LeafDataset(A_DIR / "val", plain_transform))
    if CAMERA_DIR.exists() and len(list(CAMERA_DIR.rglob("*.png"))) == expected:
        print("camera_val: already generated")
        return
    set_seed(0)
    augment = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.RandomHorizontalFlip(),
    ])
    for class_dir in sorted((A_DIR / "val").iterdir()):
        if not class_dir.is_dir():
            continue
        (CAMERA_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)
        for image_path in sorted(class_dir.iterdir()):
            image = Image.open(image_path).convert("RGB")
            for copy in range(copies):
                augment(image).save(CAMERA_DIR / class_dir.name / f"{image_path.stem}_c{copy}.png")
    print(f"camera_val: generated {expected} images with seed 0")


def export(name, seed):
    """Train one variant and write the phone files exactly as Task 2e did."""
    model = run(name, seed)
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), ROOT / "leaf_model.onnx",
                      input_names=["image"], output_names=["logits"], external_data=False)

    import onnx
    import onnxruntime
    classes = LeafDataset(A_DIR / "train", plain_transform).classes
    model_info = {"classes": classes, "input_size": 96, "mean": MEAN, "std": STD}
    onnx_model = onnx.load(ROOT / "leaf_model.onnx")
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, ROOT / "leaf_model.onnx")
    (ROOT / "model_info.json").write_text(json.dumps(model_info))

    check_path = A_DIR / "val" / classes[0] / sorted((A_DIR / "val" / classes[0]).iterdir())[0].name
    Image.open(check_path).save(ROOT / "check_image.jpg", quality=95)
    x = plain_transform(Image.open(ROOT / "check_image.jpg").convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
    session = onnxruntime.InferenceSession(str(ROOT / "leaf_model.onnx"))
    raw = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0][0]
    difference = np.abs(logits.numpy() - raw).max()
    print(f"exported {name} seed {seed}; PyTorch vs ONNX logit difference {difference:.2e}")

    probabilities = dict(zip(classes, logits.softmax(dim=0).tolist()))
    (ROOT / "check_expected.json").write_text(json.dumps(probabilities, indent=1))
    print(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download B and H, build camera_val")
    parser.add_argument("--run", choices=VARIANTS, help="train one variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", choices=VARIANTS, help="train one variant and write the phone files")
    parser.add_argument("--summary", action="store_true", help="print the comparison tables")
    parser.add_argument("--list", action="store_true", help="print the variant table")
    arguments = parser.parse_args()

    if arguments.prepare:
        prepare()
    if arguments.run:
        run(arguments.run, arguments.seed)
    if arguments.export:
        export(arguments.export, arguments.seed)
    if arguments.summary:
        summary()
    if arguments.list:
        for name, cfg in VARIANTS.items():
            print(f"{name:10s} parent {cfg['parent']:4s} train {cfg['train_set']:3s} "
                  f"{cfg['epochs']:2d} ep  {cfg['change']}")


if __name__ == "__main__":
    main()
