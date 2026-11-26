# Facial Emotion Recognition (FER2013) with a CNN

A convolutional neural network, built from scratch in PyTorch, that classifies a face into one of 7 emotions: **angry, disgust, fear, happy, neutral, sad, surprise**. Trained on the FER2013 dataset (48x48 grayscale faces).

The repo includes the full training pipeline, evaluation (classification report and confusion matrix), and a prediction script that detects faces in any photo and labels each one.

## Results

Trained for 40 epochs on a free Colab T4 GPU (about 21 s per epoch).

| Metric | Value |
|---|---|
| Test accuracy | **68.5%** |
| Best validation accuracy | 69.5% |
| Macro F1 | 0.665 |
| Parameters | 5.87M |

Per-class results on the 7,178 test images:

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| angry | 0.588 | 0.646 | 0.616 |
| disgust | 0.635 | 0.658 | 0.646 |
| fear | 0.577 | 0.463 | 0.514 |
| happy | 0.876 | 0.877 | 0.876 |
| neutral | 0.600 | 0.693 | 0.643 |
| sad | 0.585 | 0.523 | 0.552 |
| surprise | 0.786 | 0.828 | 0.807 |

Happy and surprise are recognised best. Fear and sad are the hardest, and are most often confused with angry, neutral and each other. FER2013 labels are known to be noisy, which limits the ceiling for this kind of model.

![Training curves](outputs/training_curves.png)
![Confusion matrix](outputs/confusion_matrix.png)

## Model

- 3 convolutional blocks (64, 128, 256 filters), each with two 3x3 convs, BatchNorm, ReLU, MaxPool and Dropout
- Fully connected head (512 units) with BatchNorm and Dropout
- Augmentation: horizontal flip, small rotation, random resized crop
- Class-weighted cross-entropy (softened inverse frequency, because "disgust" has only ~400 training images) with label smoothing 0.1
- AdamW optimizer with a OneCycle learning-rate schedule, early stopping, best checkpoint kept by validation accuracy

## Dataset

FER2013, image-folder version: https://www.kaggle.com/datasets/msambare/fer2013

Download and unzip it so the layout is:

```
fer2013/
    train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    test/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

The dataset is not included in this repo.

## Setup

```
pip install -r requirements.txt
pip install opencv-python     # needed only for predict.py
```

## Train

```
python train.py --data_dir fer2013 --epochs 40
```

A GPU is strongly recommended (about 20 s per epoch on a Colab T4 versus about 18 minutes per epoch on a laptop CPU). Outputs (model, curves, confusion matrix, report) are saved to `outputs/`.

## Predict

```
python predict.py face.jpg
```

The script detects each face (OpenCV Haar cascade), crops it, and prints the top 3 emotions with probabilities. An annotated copy of the image is saved as `face_pred.jpg`. Use `--no_crop` if the image is already a tight 48x48 face crop.

For best results use a clear, well-lit, front-facing photo.

## Project structure

```
train.py            model, training loop, evaluation
predict.py          face detection + emotion prediction
requirements.txt
outputs/            trained weights, curves, confusion matrix, report
```

## Limitations and ideas

- Trained on small grayscale faces, so accuracy drops with heavy blur, extreme angles or occlusion.
- Possible improvements: transfer learning from a pretrained backbone (e.g. ResNet18), the cleaner FER+ labels, stronger augmentation, test-time augmentation.
