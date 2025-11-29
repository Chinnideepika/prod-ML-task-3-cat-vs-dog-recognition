# 🐱🐶 PRODI GY_ML_03 – Cats vs Dogs Classification Using HOG + SVM

This project is my submission for **Task 03** of the **Prodigy InfoTech Machine Learning Internship**.

The goal of this task is to classify images of **Cats** and **Dogs** using a **classical Machine Learning pipeline** involving **HOG (Histogram of Oriented Gradients)** for feature extraction and **Support Vector Machine (SVM)** for classification.

---

## 📊 Dataset
The dataset used is the **Kaggle Cats and Dogs Dataset** (Microsoft Research), which contains:

- 12,500 cat images  
- 12,500 dog images  
- Real-world, unfiltered data (including some corrupted images)

---

## 🧠 Methodology

### 🔹 Step 1: Load and Preprocess Images  
- Converted each image to **grayscale**  
- Resized images to **64×64** for consistency  
- Handled corrupted files by skipping unreadable images

### 🔹 Step 2: Feature Extraction with HOG  
Extracted **Histogram of Oriented Gradients** features, which capture:
- Edges  
- Shapes  
- Gradient directions  

These features allow an SVM to classify images without deep learning.

### 🔹 Step 3: Train-Test Split  
Split the dataset into:
- **80% training**
- **20% testing**

### 🔹 Step 4: Train SVM Classifier  
Used **scikit-learn SVM (Support Vector Machine)** with hyperparameter tuning.  
Saved the final model as:

```
svm_model.joblib
```

### 🔹 Step 5: Prediction Script  
Implemented easy-to-use prediction mode:

```bash
%run hog_svm_cats_dogs.py --mode predict --model_path "output/svm_model.joblib" --image_path "test.jpg"
```

---

## 📈 Output

- Extracted HOG feature vectors for all images  
- Trained SVM classifier  
- Generated predictions for new cat/dog images  
- Model saved in `output/`

---

## 📁 Files in this Repository

- `hog_svm_cats_dogs.py` – Main ML pipeline (training + prediction)  
- `output/svm_model.joblib` – Saved trained model  
- `README.md` – Documentation  

---

## 🏁 Conclusion  
This project demonstrates how **classical machine learning techniques** like HOG + SVM can still perform well on image classification tasks.

Key insights include:

- HOG captures **edges and shapes**, which are useful for distinguishing cats vs dogs  
- SVM performs strongly on **structured feature vectors**  
- Complete ML workflow implemented: preprocessing → feature extraction → training → saving → prediction  

This task helped me strengthen my understanding of **traditional computer vision** before moving into deep learning.

---

### 🚀 Developed by Deepika  
Prodigy InfoTech – Machine Learning Internship  

