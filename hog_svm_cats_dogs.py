# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 04:08:42 2025

@author: Deepika
"""

#!/usr/bin/env python3
"""
hog_svm_cats_dogs.py

A beginner-friendly script to train an SVM on HOG features for Cats vs Dogs.

Usage examples:
  # Train
  python hog_svm_cats_dogs.py --data_dir /path/to/dataset --output_dir ./output --mode train

  # Predict single image (after training or with provided svm_model.joblib)
  python hog_svm_cats_dogs.py --mode predict --model_path ./output/svm_model.joblib --image_path /path/to/image.jpg

Expect dataset layout (common): 
  data_dir/
    cats/   (cat images)
    dogs/   (dog images)
If your folders are named differently, pass --cats_dir and --dogs_dir.
"""

import os
import argparse
from pathlib import Path
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# -------------------------
# Helpers
# -------------------------
def gather_image_paths(cats_dir, dogs_dir, max_per_class=None):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    def list_images(folder):
        p = Path(folder)
        if not p.exists():
            return []
        files = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
        return files[:max_per_class] if max_per_class else files
    cats = list_images(cats_dir)
    dogs = list_images(dogs_dir)
    paths = cats + dogs
    labels = [0]*len(cats) + [1]*len(dogs)  # 0: cat, 1: dog
    return paths, labels

def load_and_preprocess(path, image_size=(64,64)):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    # resize to fixed shape (important for HOG consistency)
    img_resized = transform.resize(img, image_size, anti_aliasing=True)
    return img_resized

def extract_hog_features(paths, pixels_per_cell=(8,8), cells_per_block=(2,2),
                         orientations=9, image_size=(64,64), visualize=False):
    features = []
    hog_imgs = []  # optional, when visualize=True
    for p in tqdm(paths, desc="Extracting HOG"):
        try:
            img = load_and_preprocess(p, image_size=image_size)
            fd = hog(img,
                     orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block,
                     block_norm='L2-Hys',
                     visualize=visualize,
                     feature_vector=True)
            features.append(fd)
        except Exception as e:
            # skip unreadable images but warn
            print(f"Warning: couldn't process {p}: {e}")
    X = np.array(features)
    return X

# -------------------------
# Train / Evaluate
# -------------------------
def train_svm(X, y, output_dir, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 0.01, 0.001]
    }
    svc = SVC(probability=True)
    grid = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train_s, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['cat','dog'])
    cm = confusion_matrix(y_test, y_pred)

    print("Best params:", grid.best_params_)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "svm_model.joblib")
    joblib.dump({'model': best, 'scaler': scaler}, model_path)
    print("Saved model+scaler to:", model_path)
    return model_path

# -------------------------
# Predict single image
# -------------------------
def predict_single(image_path, model_path, image_size=(64,64)):
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']
    img = load_and_preprocess(image_path, image_size=image_size)
    fd = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
             block_norm='L2-Hys', visualize=False, feature_vector=True)
    Xs = scaler.transform([fd])
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0]
    label = 'dog' if pred==1 else 'cat'
    confidence = prob[pred]
    print(f"Prediction: {label} (confidence {confidence:.3f})")
    return label, confidence

# -------------------------
# CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="HOG + SVM Cats vs Dogs")
    parser.add_argument('--data_dir', type=str, help='Root folder containing cats/ and dogs/')
    parser.add_argument('--cats_dir', type=str, default=None, help='Path to cats folder (overrides)')
    parser.add_argument('--dogs_dir', type=str, default=None, help='Path to dogs folder (overrides)')
    parser.add_argument('--max_per_class', type=int, default=1000, help='Limit images per class (useful on laptop)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where to save model')
    parser.add_argument('--mode', type=str, choices=['train','predict'], default='train', help='train or predict')
    parser.add_argument('--model_path', type=str, help='Path to saved model for prediction')
    parser.add_argument('--image_path', type=str, help='Image path for single prediction')
    parser.add_argument('--image_size', type=int, default=64, help='Resize side length (square)')
    args = parser.parse_args()

    if args.mode == 'train':
        # determine cats/dogs folders
        if args.cats_dir and args.dogs_dir:
            cats_dir = args.cats_dir
            dogs_dir = args.dogs_dir
        else:
            dd = Path(args.data_dir) if args.data_dir else None
            if not dd:
                raise ValueError("Please supply --data_dir or both --cats_dir and --dogs_dir")
            # try usual names
            if (dd/'cats').exists() and (dd/'dogs').exists():
                cats_dir = str(dd/'cats'); dogs_dir = str(dd/'dogs')
            else:
                # fallback: take first two subfolders
                subs = [p for p in dd.iterdir() if p.is_dir()]
                if len(subs) >= 2:
                    cats_dir = str(subs[0]); dogs_dir = str(subs[1])
                else:
                    raise ValueError("Couldn't infer cats/dogs folders. Use --cats_dir and --dogs_dir")

        print("Using cats:", cats_dir)
        print("Using dogs:", dogs_dir)
        paths, labels = gather_image_paths(cats_dir, dogs_dir, max_per_class=args.max_per_class)
        print(f"Found {len(paths)} images (cats+dogs). Extracting HOG features...")
        X = extract_hog_features(paths, image_size=(args.image_size,args.image_size))
        y = np.array(labels[:len(X)])  # ensure label length matches features (skips bad images)
        print("Feature matrix shape:", X.shape)
        train_svm(X, y, args.output_dir)

    elif args.mode == 'predict':
        if not args.model_path or not args.image_path:
            raise ValueError("For predict mode, supply --model_path and --image_path")
        predict_single(args.image_path, args.model_path, image_size=(args.image_size,args.image_size))

if __name__ == "__main__":
    main()
