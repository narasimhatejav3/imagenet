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


### **4. T4 GPU Optimizations**
- **Mixed Precision (AMP)**: 2-3x faster training
- **Batch Size**: 256 (optimized for 16GB memory)
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
📊 Epoch 1/30 Summary:
   Train Loss: 4.5861 | Train Acc: 2.77%
   Val Loss: 4.2938 | Val Acc: 6.90%
   LR: 6.90e-03

✅ New best model saved! Accuracy: 6.90%

Epoch 2 [Train]: 100%|██████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=4.2959, acc=6.72%, lr=1.52e-02]
Epoch 2 [Val]: 100%|███████████████████████████████████████████████████████████████| 20/20 [00:11<00:00,  1.80it/s, loss=4.3394, acc=9.08%]

📊 Epoch 2/30 Summary:
   Train Loss: 4.2959 | Train Acc: 6.72%
   Val Loss: 4.3394 | Val Acc: 9.08%
   LR: 1.52e-02

✅ New best model saved! Accuracy: 9.08%

Epoch 3 [Train]: 100%|█████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=4.0837, acc=10.31%, lr=2.80e-02]
Epoch 3 [Val]: 100%|██████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.06it/s, loss=3.7933, acc=16.04%]

📊 Epoch 3/30 Summary:
   Train Loss: 4.0837 | Train Acc: 10.31%
   Val Loss: 3.7933 | Val Acc: 16.04%
   LR: 2.80e-02

✅ New best model saved! Accuracy: 16.04%

Epoch 4 [Train]:  16%|███████▌                                      | 81/495 [01:21<06:48,  1.01it/s, loss=3.9953, acc=12.18%, lr=3.04e-02]Epoch 4 [Train]:  17%|███████▌                                      | 82/495 [01:22<06:47,  1.01it/s, loss=3.9990, acc=12.12%, lr=3.05e-02]Epoch 4 [Train]:  99%|████████████████████████████████Epoch 4 [Train]: 100%|█████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.9162, acc=14.07%, lr=4.37e-02]
Epoch 4 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.94it/s, loss=3.6021, acc=21.74%]

📊 Epoch 4/30 Summary:
   Train Loss: 3.9162 | Train Acc: 14.07%
   Val Loss: 3.6021 | Val Acc: 21.74%
   LR: 4.37e-02

✅ New best model saved! Accuracy: 21.74%

Epoch 5 [Train]: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.7278, acc=18.53%, lr=6.04e-02]
Epoch 5 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, loss=3.3724, acc=24.42%]

📊 Epoch 5/30 Summary:
   Train Loss: 3.7278 | Train Acc: 18.53%
   Val Loss: 3.3724 | Val Acc: 24.42%
   LR: 6.04e-02

✅ New best model saved! Accuracy: 24.42%

Epoch 6 [Train]: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [08:11<00:00,  1.01it/s, loss=3.5914, acc=21.61%, lr=7.60e-02]
Epoch 6 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.97it/s, loss=3.1700, acc=29.10%]

📊 Epoch 6/30 Summary:
   Train Loss: 3.5914 | Train Acc: 21.61%
   Val Loss: 3.1700 | Val Acc: 29.10%
   LR: 7.60e-02

✅ New best model saved! Accuracy: 29.10%

Epoch 7 [Train]: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [08:11<00:00,  1.01it/s, loss=3.4330, acc=25.81%, lr=8.88e-02]
Epoch 7 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.89it/s, loss=3.0430, acc=34.70%]

📊 Epoch 7/30 Summary:
   Train Loss: 3.4330 | Train Acc: 25.81%
   Val Loss: 3.0430 | Val Acc: 34.70%
   LR: 8.88e-02

✅ New best model saved! Accuracy: 34.70%

Epoch 8 [Train]: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.2797, acc=29.65%, lr=9.71e-02]
Epoch 8 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.89it/s, loss=2.7922, acc=38.80%]

📊 Epoch 8/30 Summary:
   Train Loss: 3.2797 | Train Acc: 29.65%
   Val Loss: 2.7922 | Val Acc: 38.80%
   LR: 9.71e-02

✅ New best model saved! Accuracy: 38.80%

Epoch 9 [Train]: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.1784, acc=32.74%, lr=1.00e-01]
Epoch 9 [Val]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, loss=2.7239, acc=41.50%]

📊 Epoch 9/30 Summary:
   Train Loss: 3.1784 | Train Acc: 32.74%
   Val Loss: 2.7239 | Val Acc: 41.50%
   LR: 1.00e-01

✅ New best model saved! Accuracy: 41.50%

Epoch 10 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.0588, acc=35.65%, lr=9.94e-02]
Epoch 10 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.04it/s, loss=2.6688, acc=44.14%]

📊 Epoch 10/30 Summary:
   Train Loss: 3.0588 | Train Acc: 35.65%
   Val Loss: 2.6688 | Val Acc: 44.14%
   LR: 9.94e-02

✅ New best model saved! Accuracy: 44.14%

Epoch 11 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=3.0036, acc=37.43%, lr=9.78e-02]
Epoch 11 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.92it/s, loss=2.5939, acc=46.70%]

📊 Epoch 11/30 Summary:
   Train Loss: 3.0036 | Train Acc: 37.43%
   Val Loss: 2.5939 | Val Acc: 46.70%
   LR: 9.78e-02

✅ New best model saved! Accuracy: 46.70%

Epoch 12 [Train]:  71%|██████████████████████████████████████████████████                     | 349/495 [05:46<02:24,  1.01it/s, loss=2.8996, acc=40.03%, lr=9.60e-02]Epoch 12 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.8935, acc=40.22%, lr=9.50e-02]
Epoch 12 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.98it/s, loss=2.6388, acc=44.88%]

📊 Epoch 12/30 Summary:
   Train Loss: 2.8935 | Train Acc: 40.22%
   Val Loss: 2.6388 | Val Acc: 44.88%
   LR: 9.50e-02

✅ New best model saved! Accuracy: 50.30%

Epoch 14 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.7684, acc=43.51%, lr=8.66e-02]
Epoch 14 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.12it/s, loss=2.3858, acc=50.94%]

📊 Epoch 14/30 Summary:
   Train Loss: 2.7684 | Train Acc: 43.51%
   Val Loss: 2.3858 | Val Acc: 50.94%
   LR: 8.66e-02

✅ New best model saved! Accuracy: 50.94%

Epoch 15 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.7467, acc=44.61%, lr=8.12e-02]
Epoch 15 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.00it/s, loss=2.1588, acc=58.18%]

📊 Epoch 15/30 Summary:
   Train Loss: 2.7467 | Train Acc: 44.61%
   Val Loss: 2.1588 | Val Acc: 58.18%
   LR: 8.12e-02

✅ New best model saved! Accuracy: 58.18%

Epoch 16 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.6733, acc=46.67%, lr=7.50e-02]
Epoch 16 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.04it/s, loss=2.1544, acc=57.78%]

📊 Epoch 16/30 Summary:
   Train Loss: 2.6733 | Train Acc: 46.67%
   Val Loss: 2.1544 | Val Acc: 57.78%
   LR: 7.50e-02

Epoch 17 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.6212, acc=48.06%, lr=6.83e-02]
Epoch 17 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.10it/s, loss=2.0451, acc=61.40%]

📊 Epoch 17/30 Summary:
   Train Loss: 2.6212 | Train Acc: 48.06%
   Val Loss: 2.0451 | Val Acc: 61.40%
   LR: 6.83e-02

✅ New best model saved! Accuracy: 61.40%

Epoch 18 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.5949, acc=48.95%, lr=6.11e-02]
Epoch 18 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.06it/s, loss=1.9705, acc=63.44%]

📊 Epoch 18/30 Summary:
   Train Loss: 2.5949 | Train Acc: 48.95%
   Val Loss: 1.9705 | Val Acc: 63.44%
   LR: 6.11e-02

✅ New best model saved! Accuracy: 63.44%

Epoch 19 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.5642, acc=49.72%, lr=5.37e-02]
Epoch 19 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, loss=1.9783, acc=63.66%]

📊 Epoch 19/30 Summary:
   Train Loss: 2.5642 | Train Acc: 49.72%
   Val Loss: 1.9783 | Val Acc: 63.66%
   LR: 5.37e-02

✅ New best model saved! Accuracy: 63.66%

Epoch 20 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.4951, acc=51.66%, lr=4.62e-02]
Epoch 20 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.02it/s, loss=1.9362, acc=64.12%]

📊 Epoch 20/30 Summary:
   Train Loss: 2.4951 | Train Acc: 51.66%
   Val Loss: 1.9362 | Val Acc: 64.12%
   LR: 4.62e-02

✅ New best model saved! Accuracy: 64.12%


Epoch 21 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.4460, acc=53.07%, lr=3.89e-02]
Epoch 21 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.01it/s, loss=1.8187, acc=68.40%]

📊 Epoch 21/30 Summary:
   Train Loss: 2.4460 | Train Acc: 53.07%
   Val Loss: 1.8187 | Val Acc: 68.40%
   LR: 3.89e-02

✅ New best model saved! Accuracy: 68.40%

Epoch 22 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.4284, acc=53.99%, lr=3.17e-02]
Epoch 22 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.07it/s, loss=1.8171, acc=68.68%]

📊 Epoch 22/30 Summary:
   Train Loss: 2.4284 | Train Acc: 53.99%
   Val Loss: 1.8171 | Val Acc: 68.68%
   LR: 3.17e-02

✅ New best model saved! Accuracy: 68.68%

Epoch 23 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.3923, acc=54.91%, lr=2.50e-02]
Epoch 23 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.08it/s, loss=1.7840, acc=69.88%]

📊 Epoch 23/30 Summary:
   Train Loss: 2.3923 | Train Acc: 54.91%
   Val Loss: 1.7840 | Val Acc: 69.88%
   LR: 2.50e-02

✅ New best model saved! Accuracy: 69.88%

Epoch 24 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.3131, acc=56.95%, lr=1.88e-02]
Epoch 24 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.07it/s, loss=1.6662, acc=73.20%]

📊 Epoch 24/30 Summary:
   Train Loss: 2.3131 | Train Acc: 56.95%
   Val Loss: 1.6662 | Val Acc: 73.20%
   LR: 1.88e-02

✅ New best model saved! Accuracy: 73.20%

Epoch 25 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.3066, acc=57.38%, lr=1.33e-02]
Epoch 25 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.06it/s, loss=1.5950, acc=75.14%]

📊 Epoch 25/30 Summary:
   Train Loss: 2.3066 | Train Acc: 57.38%
   Val Loss: 1.5950 | Val Acc: 75.14%
   LR: 1.33e-02

✅ New best model saved! Accuracy: 75.14%

Epoch 26 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:09<00:00,  1.01it/s, loss=2.1845, acc=60.63%, lr=8.68e-03]
Epoch 26 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.07it/s, loss=1.5295, acc=76.94%]

📊 Epoch 26/30 Summary:
   Train Loss: 2.1845 | Train Acc: 60.63%
   Val Loss: 1.5295 | Val Acc: 76.94%
   LR: 8.68e-03

✅ New best model saved! Accuracy: 76.94%

Epoch 27 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.1847, acc=60.93%, lr=4.95e-03]
Epoch 27 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.04it/s, loss=1.5212, acc=77.98%]

📊 Epoch 27/30 Summary:
   Train Loss: 2.1847 | Train Acc: 60.93%
   Val Loss: 1.5212 | Val Acc: 77.98%
   LR: 4.95e-03

✅ New best model saved! Accuracy: 77.98%

Epoch 28 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:09<00:00,  1.01it/s, loss=2.0878, acc=63.64%, lr=2.22e-03]
Epoch 28 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.07it/s, loss=1.4553, acc=79.34%]

📊 Epoch 28/30 Summary:
   Train Loss: 2.0878 | Train Acc: 63.64%
   Val Loss: 1.4553 | Val Acc: 79.34%
   LR: 2.22e-03

✅ New best model saved! Accuracy: 79.34%


Epoch 29 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.0265, acc=65.20%, lr=5.57e-04]
Epoch 29 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.05it/s, loss=1.4304, acc=80.46%]

📊 Epoch 29/30 Summary:
   Train Loss: 2.0265 | Train Acc: 65.20%
   Val Loss: 1.4304 | Val Acc: 80.46%
   LR: 5.57e-04

✅ New best model saved! Accuracy: 80.46%

Epoch 30 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [08:10<00:00,  1.01it/s, loss=2.0419, acc=65.13%, lr=4.02e-07]
Epoch 30 [Val]: 100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.09it/s, loss=1.4263, acc=80.40%]

📊 Epoch 30/30 Summary:
   Train Loss: 2.0419 | Train Acc: 65.13%
   Val Loss: 1.4263 | Val Acc: 80.40%
   LR: 4.02e-07


🎉 Training completed! Best validation accuracy: 80.46%  
```
