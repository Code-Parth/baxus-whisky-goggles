import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import json
import faiss
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import base64

# Set page config
st.set_page_config(
    page_title="Bottle Recognition System",
    page_icon="🍾",
    layout="wide"
)

# App title and description
st.title("Bottle Recognition System")
st.markdown("Upload an image to identify the bottle!")

# Function to create a download link for a plot


def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

# Function to load model


@st.cache_resource
def load_model(model_path):
    """Load the bottle recognition model from disk"""
    try:
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load FAISS index
        index_path = os.path.join(model_path, "index.faiss")
        index = faiss.read_index(index_path)

        # Load bottle IDs
        ids_path = os.path.join(model_path, "ids.pickle")
        with open(ids_path, 'rb') as f:
            ids = pickle.load(f)

        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.pickle")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Load the feature extractor model
        feature_extractor = MobileNetV2(
            weights='imagenet', include_top=False, pooling='avg')

        return {
            'config': config,
            'index': index,
            'ids': ids,
            'metadata': metadata,
            'feature_extractor': feature_extractor
        }
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Function to extract features from an image


def extract_features(img, model):
    """Extract features from an image using the feature extractor model"""
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    features = model.predict(x, verbose=0)
    return features[0].astype('float32')

# Function to search for similar bottles with improved confidence calculation


def search_bottles(features, model_data, top_k=5):
    """Search for similar bottles in the index"""
    index = model_data['index']
    ids = model_data['ids']
    metadata = model_data['metadata']

    # Search the index
    D, I = index.search(features.reshape(1, -1), top_k)

    # First pass: calculate raw confidence scores
    results = []
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        bottle_id = ids[idx]
        info = metadata[idx].copy()  # Copy to avoid modifying the original
        info['id'] = bottle_id

        # Calculate raw confidence score (inversely proportional to distance)
        # This gives a value between 0 and 1, where closest match has highest value
        raw_confidence = float(1.0 / (1.0 + dist))

        # Store original distance for normalization
        info['distance'] = float(dist)
        info['raw_confidence'] = raw_confidence  # Store raw confidence
        info['rank'] = i + 1
        results.append(info)

    # Normalize the confidence scores across results to better use the 0-1 range
    # Get min and max distances to use for scaling
    min_dist = min(result['distance'] for result in results)
    max_dist = max(result['distance'] for result in results)

    # Normalize only if we have a range of distances
    if max_dist > min_dist:
        dist_range = max_dist - min_dist
        for result in results:
            # Invert and normalize distance to get confidence in 0-1 range
            # This maps the smallest distance to 1.0 and largest to 0.0
            result['confidence'] = float(
                1.0 - (result['distance'] - min_dist) / dist_range)
    else:
        # If all distances are the same, assign high confidence to all
        for result in results:
            result['confidence'] = 1.0

    # Clean up temporary fields
    for result in results:
        del result['distance']
        del result['raw_confidence']

    return results

# Function to clear session state


def clear_image():
    if 'current_image' in st.session_state:
        del st.session_state.current_image
    if 'results' in st.session_state:
        del st.session_state.results
    st.rerun()


# Use fixed model directory
MODEL_DIR = "model"

# Create a horizontal line to separate sections
st.markdown("---")

# Model selection section
st.header("Model Selection")

# Find all model directories
if os.path.exists(MODEL_DIR):
    model_dirs = [d for d in os.listdir(
        MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    if model_dirs:
        selected_model = st.selectbox("Select a model", model_dirs)
        model_path = os.path.join(MODEL_DIR, selected_model)

        # Load the selected model
        with st.spinner("Loading model..."):
            model_data = load_model(model_path)

        if model_data:
            # Display model information
            config = model_data['config']

            # Create a model info section that's less prominent
            with st.expander("Model Information"):
                st.write(f"Model Name: {config['model_name']}")
                st.write(f"Feature Extractor: {config['feature_extractor']}")
                st.write(f"Number of Bottles: {config['num_bottles']}")

                if 'evaluation' in config:
                    st.write(
                        f"Top-1 Accuracy: {config['evaluation']['top1_accuracy']:.2f}")
                    st.write(
                        f"Top-{config['evaluation']['top_k']} Accuracy: {config['evaluation']['topk_accuracy']:.2f}")

            # Settings for recognition
            top_k = st.slider("Number of results to show", 1, 10, 5)

            # Create a horizontal line to separate sections
            st.markdown("---")

            # Image upload section
            st.header("Upload Image")
            st.write("Upload a bottle image to identify it:")

            # Option for camera or upload
            img_source = st.radio("Select image source:", ["Upload", "Camera"])

            if img_source == "Upload":
                uploaded_file = st.file_uploader(
                    "Choose a bottle image file", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    # Display the uploaded image
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Image", width=400)

                    # Save uploaded image for reuse if needed
                    st.session_state.current_image = img

            else:  # Camera
                st.write("Take a picture of the bottle:")
                camera_img = st.camera_input("Take a picture")

                if camera_img is not None:
                    # Display and save the camera image
                    img = Image.open(camera_img)
                    st.session_state.current_image = img

            # Action buttons row
            col1, col2 = st.columns(2)

            with col1:
                # Recognize button
                recognize_btn = st.button("Identify Bottle")

            with col2:
                # Clear button
                clear_btn = st.button("Clear Image")
                if clear_btn:
                    clear_image()

            if recognize_btn and hasattr(st.session_state, 'current_image'):
                with st.spinner("Analyzing image..."):
                    # Extract features
                    features = extract_features(
                        st.session_state.current_image, model_data['feature_extractor'])

                    # Search for similar bottles
                    results = search_bottles(features, model_data, top_k)

                    # Store results in session state
                    st.session_state.results = results

                    # Display success message
                    st.success("Analysis complete!")

            elif recognize_btn and not hasattr(st.session_state, 'current_image'):
                st.error("Please upload an image or take a picture first.")

            # Create a horizontal line to separate sections
            st.markdown("---")

            # Results section
            st.header("Recognition Results")

            if 'results' in st.session_state and st.session_state.results:
                results = st.session_state.results

                # Create a horizontal bar chart of confidence scores
                fig, ax = plt.subplots(figsize=(10, 5))

                # Extract bottle names and confidence values
                bottle_names = []
                confidence_values = []

                for i, result in enumerate(results):
                    # Get bottle name from metadata
                    name_col = config.get('name_column', 'name')
                    # Try to get the name column, fall back to id if not available
                    if name_col in result:
                        bottle_name = result[name_col]
                    else:
                        bottle_name = f"Bottle {result['id']}"

                    # Truncate long names
                    if len(bottle_name) > 30:
                        bottle_name = bottle_name[:27] + "..."

                    bottle_names.append(bottle_name)
                    confidence_values.append(result['confidence'])

                # Create bar chart
                y_pos = np.arange(len(bottle_names))
                bars = ax.barh(y_pos, confidence_values, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(bottle_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Confidence Score (0-1)')
                ax.set_title('Top Matches')
                # Set x-axis range to 0-1 with a little margin
                ax.set_xlim(0, 1.05)

                # Add score labels as decimals (0-1 range)
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f"{confidence_values[i]:.2f}", va='center')

                # Display the chart
                st.pyplot(fig)

                # Provide a download link for the chart
                st.markdown(get_image_download_link(
                    fig, "bottle_matches.png", "Download Chart"), unsafe_allow_html=True)

                # Display detailed results in a table
                st.subheader("Detailed Results")

                # Determine common fields across all results to create table columns
                # Start with priority fields
                priority_fields = ['rank', 'id', 'name', 'brand', 'type',
                                   'category', 'price', 'volume', 'confidence']

                # Filter to only include fields that actually exist in results
                available_fields = set()
                for result in results:
                    available_fields.update(result.keys())

                # Create ordered list of fields to display
                display_fields = [
                    f for f in priority_fields if f in available_fields]

                # Add any other fields that exist but weren't in priority list
                other_fields = [
                    f for f in available_fields if f not in priority_fields]
                display_fields.extend(other_fields)

                # Prepare data for the table
                table_data = []
                for result in results:
                    row = {}
                    for field in display_fields:
                        if field in result:
                            value = result[field]
                            row[field] = value
                        else:
                            row[field] = ""
                    table_data.append(row)

                # Create DataFrame for the table
                df = pd.DataFrame(table_data)

                # Format confidence to 2 decimal places for display
                if 'confidence' in df.columns:
                    df['confidence'] = df['confidence'].map(
                        lambda x: f"{float(x):.2f}")

                # Rename columns to be more user-friendly
                renamed_columns = {col: col.title() for col in df.columns}
                df = df.rename(columns=renamed_columns)

                # Display the table
                st.dataframe(df, use_container_width=True)

            else:
                st.info("Upload an image and click 'Identify Bottle' to see results")
        else:
            st.error(
                "Failed to load the selected model. Please check the model path.")
    else:
        st.warning(
            f"No models found in the 'model' directory. Please train a model first using the training tool.")
else:
    st.warning(
        f"Model directory 'model' does not exist. Please run the training tool first to create models.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Bottle Recognition System © 2025</div>",
            unsafe_allow_html=True)
