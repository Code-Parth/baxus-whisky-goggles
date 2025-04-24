import os
import pickle
import faiss
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# === CONFIG ===
INDEX_PATH   = 'bottle_index.faiss'
IDS_PATH     = 'bottle_ids.pkl'
META_CSV     = 'bottle_metadata.csv'
TOP_K        = 5

st.set_page_config(page_title="Bottle Identifier", layout="wide")

@st.cache_resource
def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH, 'rb') as f:
        ids = pickle.load(f)
    meta = pd.read_csv(META_CSV)
    return idx, ids, meta

@st.cache_resource
def get_model():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    x   = image.img_to_array(img)
    x   = preprocess_input(np.expand_dims(x,axis=0))
    feats = model.predict(x)
    return feats[0].astype('float32')

# Load once
index, ids, meta_df = load_index()
model = get_model()

st.title("🍾 Bottle Recognition")
st.write("Upload a bottle image and see the top matches from the dataset.")

uploaded = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])
if uploaded:
    # Save temp file
    tmp_path = os.path.join("tmp_query." + uploaded.name.split('.')[-1])
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    st.image(tmp_path, caption="Query Image", use_container_width=True)

    # Query
    qf = extract_features(tmp_path, model).reshape(1,-1)
    D, I = index.search(qf, TOP_K)

    results = []
    for dist, idx in zip(D[0], I[0]):
        bid = ids[idx]
        row = meta_df[meta_df['id']==bid].iloc[0].to_dict()
        confidence = 1.0 / (1.0 + dist)
        entry = {
            'id': bid,
            'distance': float(dist),
            'confidence': confidence
        }
        # include any available metadata fields
        for fld in ['name','price','brand','type']:
            if fld in row:
                entry[fld] = row[fld]
        results.append(entry)

    if results:
        df = pd.DataFrame(results)
        st.subheader("Top Matches")
        st.table(df)
    else:
        st.error("No matches found.")

    # Clean up
    os.remove(tmp_path)
