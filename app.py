
# import streamlit as st
# import numpy as np
# import pickle
# from PIL import Image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image

# # =========================
# # Load Saved Data
# # =========================
# features = np.load("features.npy")
# images = np.load("images.npy")
# labels = np.load("labels.npy")

# with open("knn_model.pkl", "rb") as f:
#     knn = pickle.load(f)

# # Load ResNet50 (for query image feature extraction)
# resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))

# # =========================
# # Helper Function
# # =========================
# def extract_features(img):
#     img = img.resize((224,224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     feat = resnet.predict(img_array, verbose=0)
#     return feat

# def recommend(img, top_k=5):
#     feat = extract_features(img)
#     distances, indices = knn.kneighbors(feat, n_neighbors=top_k)
#     return indices[0]

# # =========================
# # Streamlit UI
# # =========================
# st.title("👗 Fashion Recommender System")
# st.write("Upload a picture and get **5 similar fashion products** recommended!")

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

# if uploaded_file is not None:
#     query_img = Image.open(uploaded_file).convert("RGB")
#     st.image(query_img, caption="Uploaded Image", width=250)

#     st.write("🔍 Finding similar products...")
#     indices = recommend(query_img, top_k=5)

#     st.subheader("Recommended Products:")
#     # cols = st.columns(5)
#     cols = st.columns(min(5, len(indices)))

#     for i, idx in enumerate(indices):
#         with cols[i]:
#             st.image(images[idx], caption=f"Item {i+1}", use_container_width=True)
            
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# =========================
# Load Saved Data
# =========================
features = np.load("features.npy")
images = np.load("images.npy")
labels = np.load("labels.npy")   # product categories/labels

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# Load ResNet50 (for query image feature extraction)
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))

# =========================
# Helper Function
# =========================
def extract_features(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feat = resnet.predict(img_array, verbose=0)
    return feat

def recommend(img, top_k=5, pool_k=15):
    # Extract features of uploaded image
    feat = extract_features(img)
    distances, indices = knn.kneighbors(feat, n_neighbors=pool_k)

    # Take label of the closest item (query category)
    query_label = labels[indices[0][0]]

    # Filter results by same category
    filtered = [idx for idx in indices[0] if labels[idx] == query_label]

    # If not enough, pad with remaining neighbors
    if len(filtered) < top_k:
        extra = [idx for idx in indices[0] if idx not in filtered]
        filtered.extend(extra)

    return filtered[:top_k]

# =========================
# Streamlit UI
# =========================
st.title("👗 Fashion Recommender System")
st.write("Upload a picture and get **5 similar fashion products** recommended!")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Uploaded Image", width=250)

    st.write("🔍 Finding similar products...")
    indices = recommend(query_img, top_k=5)

    st.subheader("Recommended Products:")
    cols = st.columns(min(5, len(indices)))

    for i, idx in enumerate(indices):
        with cols[i]:
            st.image(images[idx], caption=f"Item {i+1}", use_container_width=True)

