# ResNet50 ImageNet-100 Training - High-Level Summary

## 🎯 **Purpose**
Train ResNet50 on ImageNet-100 (100 classes) optimized for T4 GPU with automatic learning rate discovery and advanced regularization techniques.

## 🔑 **Key Components**

### **1. Data Pipeline**
- **Dataset**: HuggingFace ImageNet-100 (126k train, 5k val images)
- **Augmentation**: TrivialAugment (best-performing, parameter-free)
- **Additional**: Random crop, flip, erasing

### **2. Advanced Regularization**
- **Mixup**: Blends two images together (alpha=0.2)
- **CutMix**: Cuts patches from one image and pastes to another (alpha=1.0)
- **Dropout**: 0.2 before final layer
- **Label Smoothing**: 0.1
- **Weight Decay**: 2e-4

### **3. LR Finder (Automatic)**
- Tests learning rates from 1e-7 to 1
- Finds steepest gradient descent point
- Multiplies by 0.5 for conservative training
- **Result**: Automatically sets optimal max_lr for OneCycleLR

### **4. OneCycleLR Scheduler**
- **Phase 1 (0-30%)**: Warmup from max_lr/25 → max_lr
- **Phase 2 (30-100%)**: Annealing max_lr → max_lr/100
- **Strategy**: Cosine annealing for smooth transitions

### **5. T4 GPU Optimizations**
- **Mixed Precision (AMP)**: 2-3x faster training
- **Batch Size**: 128 (optimized for 16GB memory)
- **Persistent Workers**: Efficient data loading

## 📊 **Training Flow**

```
1. Load ImageNet-100 from HuggingFace
2. Run LR Finder (100 iterations) → finds optimal LR
3. Train 30 epochs with:
   - OneCycleLR (automatic LR scheduling)
   - Mixup/CutMix (50% of batches)
   - Mixed precision training
4. Save best model + training plots
```


## 💡 **Why This Works**
- **No manual LR tuning**: LR Finder does it automatically
- **Strong regularization**: Prevents overfitting 
- **Modern augmentation**: TrivialAugment + Mixup/CutMix
- **Efficient**: Mixed precision 


# LOGS

```
Epoch 1 [Val]: 100%|████████████████████████████████████████████| 40/40 [00:09<00:00,  4.13it/s, loss=4.3098, acc=7.38%]

📊 Epoch 1/30 Summary:
   Train Loss: 4.5501 | Train Acc: 2.98%
   Val Loss: 4.3098 | Val Acc: 7.38%
   LR: 3.07e-03

✅ New best model saved! Accuracy: 7.38%

Epoch 2 [Train]: 100%|███████████████████████████| 990/990 [07:45<00:00,  2.13it/s, loss=4.2526, acc=8.27%, lr=6.79e-03]
Epoch 2 [Val]: 100%|███████████████████████████████████████████| 40/40 [00:09<00:00,  4.22it/s, loss=3.5863, acc=21.18%]

📊 Epoch 2/30 Summary:
   Train Loss: 4.2526 | Train Acc: 8.27%
   Val Loss: 3.5863 | Val Acc: 21.18%
   LR: 6.79e-03

✅ New best model saved! Accuracy: 21.18%

Epoch 3 [Train]: 100%|██████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=3.6505, acc=21.59%, lr=1.25e-02]
Epoch 3 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.24it/s, loss=2.3934, acc=51.68%]

📊 Epoch 3/30 Summary:
   Train Loss: 3.6505 | Train Acc: 21.59%
   Val Loss: 2.3934 | Val Acc: 51.68%
   LR: 1.25e-02

✅ New best model saved! Accuracy: 51.68%

Epoch 4 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=2.7942, acc=43.89%, lr=1.95e-02]
Epoch 4 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.02it/s, loss=1.6172, acc=74.64%]

📊 Epoch 4/30 Summary:
   Train Loss: 2.7942 | Train Acc: 43.89%
   Val Loss: 1.6172 | Val Acc: 74.64%
   LR: 1.95e-02

✅ New best model saved! Accuracy: 74.64%

Epoch 5 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=2.3873, acc=55.37%, lr=2.69e-02]
Epoch 5 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.07it/s, loss=1.4559, acc=78.92%]

📊 Epoch 5/30 Summary:
   Train Loss: 2.3873 | Train Acc: 55.37%
   Val Loss: 1.4559 | Val Acc: 78.92%
   LR: 2.69e-02

✅ New best model saved! Accuracy: 78.92%

Epoch 6 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=2.2106, acc=60.16%, lr=3.39e-02]
Epoch 6 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.26it/s, loss=1.3903, acc=80.64%]

📊 Epoch 6/30 Summary:
   Train Loss: 2.2106 | Train Acc: 60.16%
   Val Loss: 1.3903 | Val Acc: 80.64%
   LR: 3.39e-02

✅ New best model saved! Accuracy: 80.64%

Epoch 8 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=2.0499, acc=64.07%, lr=4.33e-02]
Epoch 8 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.29it/s, loss=1.3369, acc=83.32%]

📊 Epoch 8/30 Summary:
   Train Loss: 2.0499 | Train Acc: 64.07%
   Val Loss: 1.3369 | Val Acc: 83.32%
   LR: 4.33e-02

✅ New best model saved! Accuracy: 83.32%

Epoch 9 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=2.0238, acc=64.79%, lr=4.46e-02]
Epoch 9 [Val]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.15it/s, loss=1.2790, acc=83.90%]

📊 Epoch 9/30 Summary:
   Train Loss: 2.0238 | Train Acc: 64.79%
   Val Loss: 1.2790 | Val Acc: 83.90%
   LR: 4.46e-02

✅ New best model saved! Accuracy: 83.90%

Epoch 10 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=1.9994, acc=65.27%, lr=4.43e-02]
Epoch 10 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.27it/s, loss=1.3119, acc=83.24%]

📊 Epoch 10/30 Summary:
   Train Loss: 1.9994 | Train Acc: 65.27%
   Val Loss: 1.3119 | Val Acc: 83.24%
   LR: 4.43e-02

Epoch 11 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=1.9848, acc=65.68%, lr=4.36e-02]
Epoch 11 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.22it/s, loss=1.3088, acc=83.04%]


📊 Epoch 11/30 Summary:
   Train Loss: 1.9848 | Train Acc: 65.68%
   Val Loss: 1.3088 | Val Acc: 83.04%
   LR: 4.36e-02

Epoch 12 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=1.9504, acc=66.59%, lr=4.24e-02]
Epoch 12 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.22it/s, loss=1.2896, acc=84.00%]

📊 Epoch 12/30 Summary:
   Train Loss: 1.9504 | Train Acc: 66.59%
   Val Loss: 1.2896 | Val Acc: 84.00%
   LR: 4.24e-02

✅ New best model saved! Accuracy: 84.00%

Epoch 13 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 990/990 [07:46<00:00,  2.12it/s, loss=1.9594, acc=66.21%, lr=4.07e-02]
Epoch 13 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.31it/s, loss=1.2715, acc=84.30%]

📊 Epoch 13/30 Summary:
   Train Loss: 1.9594 | Train Acc: 66.21%
   Val Loss: 1.2715 | Val Acc: 84.30%
   LR: 4.07e-02

✅ New best model saved! Accuracy: 84.30%   
```
