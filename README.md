##   Fashion Recommender System

A deep learning–based fashion product recommender built with ResNet50 + KNN.
Users can upload a fashion product image, and the system recommends visually similar items from a dataset of fashion product images (Kaggle dataset).

## Features

🔍 Image-based Recommendation – Upload an image and get 5 similar products.

🧠 ResNet50 Pretrained Model – Extracts 2048-d feature embeddings from images.

📊 KNN (Cosine Similarity) – Finds the most visually similar items.

📈 Evaluation Metrics – Hit Rate, Precision, Recall, Top-K Accuracy.

🎨 Visualizations – t-SNE clustering, distance distribution, accuracy plots.

🌐 Streamlit App – Interactive web UI for testing recommendations.


## 📂 Project Structure


project/

│── project.ipynb # upyter notebook (data prep, feature extraction, training, evaluation)

│── app.py # Streamlit app for interactive demo

│── features.npy # Extracted ResNet50 features

│── images.npy # Preprocessed image data

│── labels.npy # Encoded category labels

│── knn_model.pkl # Trained KNN model

│── README.md # Project documentation

## 📊 Dataset

Source: [Fashion Product Images (Small) – Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)


Size: ~44k product images (Apparel, Footwear, Accessories, etc.)

Used: 2000 samples (for memory efficiency in Colab).

## 🛠️ How It Works

## 1. Feature Extraction

Images → ResNet50 (pretrained on ImageNet) → 2048-d embeddings.

Global Average Pooling applied to reduce final feature maps.

## 2.KNN Similarity Search

KNN with cosine distance finds the closest items.

Distance-weighted voting ensures stable recommendations.

## 3.Recommendation Filtering

Ensures recommended items belong to the same category.

Pads with extra neighbors if fewer results found.

## 4.Interactive Demo

Upload an image in Streamlit.

System shows query + top-5 recommendations.



## 📈 Results

Top-1 Accuracy: ~97.7%

Top-3 Accuracy: ~99.1%

Top-5 Accuracy: ~99.3%

Top-10 Accuracy: ~99.5%

## Example Visualizations:

📊 Top-K Accuracy Bar Chart

🔵 t-SNE Feature Clustering

📉 Distance Histogram (Cosine similarity distribution)

## 📌 Future Improvements

🔹 Fine-tune ResNet50 on fashion dataset (instead of ImageNet only).

🔹 Add multi-label recommendations (color, brand, style).

🔹 Use ANN (Approximate Nearest Neighbor) for faster search on large datasets.

🔹 Deploy on cloud (Heroku/Streamlit Cloud).


