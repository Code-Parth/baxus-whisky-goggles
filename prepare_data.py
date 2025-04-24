import os
import pickle
import faiss
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# === CONFIGURATION ===
CSV_PATH = '501-Bottle-Dataset.csv'
URL_COL = 'image_url'
ID_COL = 'id'
IMAGE_DIR = 'bottle_images'
INDEX_PATH = 'bottle_index.faiss'
IDS_PATH = 'bottle_ids.pkl'
META_OUT_CSV = 'bottle_metadata.csv'
EVAL_OUT_CSV = 'evaluation_results.csv'
TOP_K = 5

os.makedirs(IMAGE_DIR, exist_ok=True)


def main():
    # === 1) Load metadata ===
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)

    # Clean and preprocess the dataframe if needed
    # Handle missing values and ensure required columns exist
    required_cols = [ID_COL, URL_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in CSV")

    # Save clean metadata
    df.to_csv(META_OUT_CSV, index=False)
    print(f"✓ Saved metadata to {META_OUT_CSV}")

    # === 2) Download images ===
    print("Downloading images...")
    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get(URL_COL)
        bid = row.get(ID_COL)
        if pd.isna(url) or pd.isna(bid):
            continue

        url = url.strip()
        if not url.lower().startswith(('http://', 'https://')):
            continue

        ext = os.path.splitext(url)[1].split('?')[0] or '.jpg'
        fn = f"{int(bid)}{ext}"
        out = os.path.join(IMAGE_DIR, fn)

        if os.path.exists(out):
            success_count += 1
            continue

        try:
            r = requests.get(url, stream=True, timeout=10)
            r.raise_for_status()
            with open(out, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            success_count += 1
        except Exception as e:
            print(f"⚠️ Failed {url}: {e}")

    print(f"✓ Downloaded {success_count} images to {IMAGE_DIR}")

    # === 3) Feature extractor ===
    print("Loading MobileNetV2...")
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    print("✓ Model loaded")

    # === 4) Build FAISS index ===
    print("Extracting features & building FAISS index...")
    features, ids = [], []
    files_processed = 0
    failed_files = []

    for fn in tqdm(os.listdir(IMAGE_DIR)):
        try:
            img_path = os.path.join(IMAGE_DIR, fn)
            bid = int(os.path.splitext(fn)[0])
            feats = extract_features(img_path, model)
            features.append(feats)
            ids.append(bid)
            files_processed += 1
        except Exception as e:
            print(f"⚠️ Error processing {fn}: {e}")
            failed_files.append(fn)

    if not features:
        raise ValueError("No valid images found or processed!")

    features = np.stack(features)
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)
    print(f"✓ Indexed {index.ntotal} images with {d} dimensions")

    # Save index & ids
    faiss.write_index(index, INDEX_PATH)
    with open(IDS_PATH, 'wb') as f:
        pickle.dump(ids, f)
    print(f"✓ Saved index to {INDEX_PATH} and IDs to {IDS_PATH}")

    # === 5) Evaluate overall accuracy ===
    print("Evaluating overall accuracy...")
    evaluate_model(model, index, ids, df, IMAGE_DIR,
                   ID_COL, URL_COL, TOP_K, EVAL_OUT_CSV)


def extract_features(img_path, model):
    """Extract features from an image using MobileNetV2"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    feats = model.predict(x, verbose=0)  # Suppress verbose output
    return feats[0].astype('float32')


def evaluate_model(model, index, ids, df, image_dir, id_col, url_col, top_k, eval_out_csv):
    """Evaluate model accuracy and save results"""
    correct_top1 = 0
    correct_topk = 0
    results = []

    for fn in tqdm(os.listdir(image_dir)):
        try:
            true_id = int(os.path.splitext(fn)[0])
            img_path = os.path.join(image_dir, fn)

            # Extract features and search
            qf = extract_features(img_path, model).reshape(1, -1)
            D, I = index.search(qf, top_k)
            preds = [ids[i] for i in I[0]]

            # Check if predictions match the true ID
            hit1 = (preds[0] == true_id)
            hitk = (true_id in preds)
            correct_top1 += int(hit1)
            correct_topk += int(hitk)

            # Get metadata for the true bottle
            meta_row = df[df[id_col] == true_id]
            if not meta_row.empty:
                meta = meta_row.iloc[0].to_dict()
                results.append({
                    'filename': fn,
                    'true_id': true_id,
                    'pred_top1': preds[0],
                    'hit_top1': hit1,
                    f'preds_top{top_k}': preds,
                    f'hit_top{top_k}': hitk,
                    'distance_top1': float(D[0][0]),
                    **{k: meta[k] for k in meta if k not in (id_col, url_col)}
                })
        except Exception as e:
            print(f"⚠️ Error evaluating {fn}: {e}")

    total = len(results)
    top1_acc = correct_top1/total if total else 0
    topk_acc = correct_topk/total if total else 0

    print(f"Evaluated {total} images")
    print(f"Top-1 Accuracy: {correct_top1}/{total} = {top1_acc:.2%}")
    print(f"Top-{top_k} Accuracy: {correct_topk}/{total} = {topk_acc:.2%}")

    # Plot accuracy results
    plot_accuracy_results(top1_acc, topk_acc, top_k)

    # Save detailed results
    res_df = pd.DataFrame(results)
    res_df.to_csv(eval_out_csv, index=False)
    print(f"✓ Saved evaluation results to {eval_out_csv}")


def plot_accuracy_results(top1_acc, topk_acc, top_k):
    """Visualize accuracy results"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Top-1 Accuracy', f'Top-{top_k} Accuracy']
    values = [top1_acc, topk_acc]

    ax.bar(metrics, values, color=['#3498db', '#2ecc71'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Bottle Recognition Performance')

    # Add percentage labels on the bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:.2%}", ha='center')

    plt.tight_layout()
    plt.savefig('accuracy_results.png')
    print("✓ Saved accuracy visualization to accuracy_results.png")
    plt.close()


if __name__ == "__main__":
    main()
