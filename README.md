# Project 2, Week 5: Scene Classification with Transfer Learning

Kaggle notebook: https://www.kaggle.com/code/shoaib1830/week-5-transfer-learning

## Overview
A ResNet-18 image classifier for six natural scene types (buildings, forest, glacier, mountain, sea, street), built with transfer learning on the Intel Image Classification dataset. I compare training only a new classifier head against also fine-tuning the last residual block.

## Dataset
- Intel Image Classification (puneet6060), 6 classes
- Splits: 11,929 train / 2,105 validation / 3,000 test
- Images resized from 150 x 150 to 224 x 224 and normalized with ImageNet statistics
- Augmentation (random resized crop, horizontal flip, colour jitter) applied to the training set only

## Model
- ResNet-18 pretrained on ImageNet
- Phase 1, feature extraction: body frozen, new 6-class head trained for 3 epochs. Trainable parameters: 3,078
- Phase 2, fine-tuning: layer4 unfrozen with learning rate 1e-4 and the head at 1e-3, for 3 more epochs. Trainable parameters: about 8.4M (8,396,806)
- The best validation epoch was kept

## Results
| Phase | Best validation accuracy |
|---|---|
| Feature extraction | 0.901 |
| Fine-tuning | 0.932 |

Test accuracy (used once): **0.935**

- Weakest class by recall: glacier, most often confused with mountain

## Most instructive error
YOUR ERROR SENTENCE: describe one misclassified image from your own error gallery (true label, predicted label, confidence, and why).

## Files
- `week-5-transfer-learning.ipynb`: notebook with outputs and written answers
- `resnet18_intel.pt`: trained model weights (44.8 MB)
- `linkedin_post.txt`: text of my LinkedIn post

## Week 6 proposal
For Project 2 I will build a crop disease classifier from leaf photographs. Plant diseases that are spotted late cost farmers part of their harvest, and a fast image-based check on a phone could help farmers and agricultural extension workers identify problems early. I will fine-tune a pretrained ResNet-18, reusing the Week 5 pipeline, on the PlantVillage dataset (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), which contains 54,306 labelled leaf images in 38 classes of healthy and diseased leaves. I expect lab-condition backgrounds to make the task easier than real field images, so I will analyse the errors and discuss this limitation honestly.
