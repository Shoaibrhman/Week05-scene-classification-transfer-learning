## Project 2, Week 5: Scene Classification with Transfer Learning

- Dataset: Intel Image Classification, 6 classes (buildings, forest, glacier, mountain, sea, street), 14034 train / 3000 test
- Model: ResNet-18 pretrained on ImageNet
- Feature extraction (3,078 trainable parameters): val accuracy 0.901
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 0.932
- Test accuracy (used once): 0.935; weakest class: glacier, confused with mountain
- Most instructive error: Glacier photos were most often predicted as mountain, because both show snow, rock and sky, and some images are genuinely ambiguous even to a human.
- Next: my own domain in Week 6: YOUR DOMAIN

### Method summary
- Images were resized to 224 x 224 and normalized with ImageNet statistics. Augmentation (random resized crop, horizontal flip, colour jitter) was applied to the training set only.
- The training set was split 85% / 15% into train and validation. The test set was used once, after all choices were made.
- Phase 1, feature extraction: the ResNet-18 body was frozen and only the new 6-class head was trained for 3 epochs (Adam, lr 1e-3).
- Phase 2, fine-tuning: layer4 was unfrozen at lr 1e-4 while the head kept lr 1e-3, for 3 more epochs. The best validation epoch was kept.
- Fine-tuning improved validation accuracy from 0.901 to 0.932 (+3.1 percentage points).

### Files
- `week-5-transfer-learning.ipynb`: full notebook with outputs and written answers
- `resnet18_intel.pt`: trained model weights (44.8 MB)
- `linkedin_post.txt`: text of my LinkedIn post

