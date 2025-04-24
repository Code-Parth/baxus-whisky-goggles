import os
import pickle
import faiss
import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# === CONFIG ===
INDEX_PATH = 'bottle_index.faiss'
IDS_PATH = 'bottle_ids.pkl'
META_CSV = 'bottle_metadata.csv'
TOP_K = 5
TMP_DIR = 'tmp_queries'

os.makedirs(TMP_DIR, exist_ok=True)
st.set_page_config(page_title="Bottle Identifier", layout="wide")

# — Cache idx + model + metadata —


@st.cache_resource
def load_index():
    try:
        idx = faiss.read_index(INDEX_PATH)
        with open(IDS_PATH, 'rb') as f:
            ids = pickle.load(f)
        meta = pd.read_csv(META_CSV)
        return idx, ids, meta
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()


@st.cache_resource
def get_model():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    feats = model.predict(x)  # Keep original without verbose parameter
    return feats[0].astype('float32')


# — Load once —
index, ids, meta_df = load_index()
model = get_model()

# — UI —
st.title("🍾 Bottle Recognition")
st.write("Select input method, then provide your image:")

# 1) Input method selector
mode = st.radio("Choose image source:", [
                "Upload from disk", "Capture from camera"])

# 2) Show only the selected widget
query_src = None
if mode == "Upload from disk":
    query_src = st.file_uploader(
        "Upload an image", type=['jpg', 'jpeg', 'png'])
else:  # Camera
    query_src = st.camera_input("Take a photo with your camera")

# 3) Process once we have an image
if query_src:
    # Use a spinner to indicate processing
    with st.spinner("Processing image..."):
        # Save to temp
        fname = query_src.name if hasattr(
            query_src, 'name') else f'camera_{int(time.time())}.jpg'
        tmp_path = os.path.join(TMP_DIR, fname)
        with open(tmp_path, 'wb') as f:
            f.write(query_src.getbuffer())

        # Show the chosen image
        st.image(tmp_path, caption="🔍 Query Image", use_container_width=True)

        # Extract features and search
        qf = extract_features(tmp_path, model).reshape(1, -1)
        D, I = index.search(qf, TOP_K)

        # Build result entries
        matches = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(ids):  # Skip invalid indices
                continue

            bid = ids[idx]
            matched_rows = meta_df[meta_df['id'] == bid]

            if matched_rows.empty:  # Skip if no matching row found
                continue

            row = matched_rows.iloc[0].to_dict()
            confidence = 1.0 / (1.0 + dist)
            entry = {
                'id': bid,
                'distance': float(dist),
                'confidence': f"{confidence:.2%}"  # Format as percentage
            }

            # Extract fields from metadata
            for fld in ('name', 'spirit_type', 'avg_msrp', 'brand'):
                if fld in row and not pd.isna(row[fld]):
                    if fld == 'avg_msrp':
                        entry['price'] = f"${row[fld]:.2f}" if row[fld] else "N/A"
                    else:
                        entry[fld] = row[fld]

            matches.append(entry)

        # Display results
        if matches:
            st.subheader("🏆 Top Matches")

            # Display using same method as original
            st.table(pd.DataFrame(matches))

            # Also show top match details
            if matches:
                st.subheader("Best Match Details")
                top_match = matches[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**ID:** {top_match.get('id')}")
                    st.markdown(f"**Name:** {top_match.get('name', 'N/A')}")
                    if 'spirit_type' in top_match:
                        st.markdown(
                            f"**Type:** {top_match.get('spirit_type')}")

                with col2:
                    st.markdown(f"**Distance:** {top_match.get('distance')}")
                    st.markdown(
                        f"**Confidence:** {top_match.get('confidence')}")
                    st.markdown(f"**Price:** {top_match.get('price', 'N/A')}")
        else:
            st.error("No matches found.")

        # Clean up
        os.remove(tmp_path)
