"""
Independent Verification Script for CIFAR-10 Student Model
Compatible with both CUDA GPU and CPU.
"""
import os
import sys
import json
import torch
from Trial3 import StudentCNN, SearchGenome, get_cifar10_loaders, evaluate_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VERIFY] Running verification on: {device}")

    genome_path = "best_student_genome.json"
    weights_path = "student_kd_best.pth"

    # Search in current directory, or common fallback paths
    search_dirs = [".", "./checkpoints_resnet50_nas", "/content/drive/MyDrive/cifar10_resnet50_nas_kd"]
    
    g_found = next((os.path.join(d, genome_path) for d in search_dirs if os.path.exists(os.path.join(d, genome_path))), None)
    w_found = next((os.path.join(d, weights_path) for d in search_dirs if os.path.exists(os.path.join(d, weights_path))), None)

    if not g_found or not w_found:
        print(f"[ERROR] Checkpoint files not found!")
        print(f"Please ensure '{genome_path}' and '{weights_path}' are in the current folder.")
        return

    with open(g_found, "r") as f:
        cfg = json.load(f)

    genome = SearchGenome(cfg["genome"])
    model = StudentCNN(genome).to(device)
    model.load_state_dict(torch.load(w_found, map_location=device))
    print(f"[SUCCESS] Loaded model weights from: {w_found}")

    print("[VERIFY] Loading untouched CIFAR-10 test set...")
    _, _, test_loader = get_cifar10_loaders(batch_size=128)

    print("[VERIFY] Computing evaluation metrics...")
    results = evaluate_metrics(model, test_loader, device)

    print("\n" + "=" * 55)
    print(f"  VERIFIED TEST ACCURACY : {results['accuracy']:.2f}%")
    print(f"  VERIFIED MACRO-F1      : {results['f1']:.2f}%")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
