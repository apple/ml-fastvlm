#!/usr/bin/env python3
"""
Benchmark script for FastVLM models.
Measures inference speed, TTFT, and basic accuracy/metrics on a dataset.
"""

import argparse
import time
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ===== Import FastVLM model loader =====
from fastvlm.models import build_model_and_transforms

# ====== Simple Metric (placeholder) =====
def simple_accuracy(predictions, references):
    """Dummy accuracy metric (for VQA-style answers)."""
    correct = sum([p.lower().strip() == r.lower().strip() for p, r in zip(predictions, references)])
    return correct / len(predictions) if predictions else 0.0


# ====== Dataset Loader (Dummy Folder Dataset) ======
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.files = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        self.transform = transform or transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), str(img_path)


# ====== Benchmark Runner ======
def run_benchmark(model_path, img_dir, device="cuda"):
    print(f"[INFO] Loading model from {model_path}")
    model, vis_processors, _ = build_model_and_transforms(
        model_path=model_path,
        device=device
    )
    model.eval()

    dataset = ImageFolderDataset(img_dir, transform=vis_processors["eval"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    total_time, total_ttft = 0.0, 0.0
    predictions, references = [], []

    for images, paths in tqdm(dataloader, desc="Benchmarking"):
        images = images.to(device)

        # Measure TTFT (Time to First Token)
        start = time.time()
        with torch.no_grad():
            output = model.generate({"image": images, "prompt": "Describe the image."})
        ttft = time.time() - start
        total_ttft += ttft

        # For demonstration, store dummy reference answer
        predictions.append(output[0])
        references.append("a photo")  # Replace with real ground truth if dataset supports it

        total_time += ttft

    avg_ttft = total_ttft / len(dataset)
    avg_latency = total_time / len(dataset)
    acc = simple_accuracy(predictions, references)

    print("\n===== Benchmark Results =====")
    print(f"Images evaluated: {len(dataset)}")
    print(f"Avg TTFT: {avg_ttft:.4f} sec")
    print(f"Avg Latency: {avg_latency:.4f} sec")
    print(f"Simple Accuracy: {acc*100:.2f}%")

    return {
        "images": len(dataset),
        "avg_ttft": avg_ttft,
        "avg_latency": avg_latency,
        "accuracy": acc
    }


# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastVLM Benchmark Script")
    parser.add_argument("--model", type=str, required=True, help="Path to FastVLM model checkpoint")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory with images to benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu/mps)")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.img-dir, device=args.device)
