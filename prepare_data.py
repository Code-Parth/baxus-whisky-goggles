import os
import pickle
import faiss
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# === CONFIGURATION ===
CSV_PATH        = '501-Bottle-Dataset.csv'
URL_COL         = 'image_url'
ID_COL          = 'id'
IMAGE_DIR       = 'bottle_images'
INDEX_PATH      = 'bottle_index.faiss'
IDS_PATH        = 'bottle_ids.pkl'
META_OUT_CSV    = 'bottle_metadata.csv'
EVAL_OUT_CSV    = 'evaluation_results.csv'
TOP_K           = 5

os.makedirs(IMAGE_DIR, exist_ok=True)

# === 1) Load metadata ===
df = pd.read_csv(CSV_PATH)
df.to_csv(META_OUT_CSV, index=False)

# === 2) Download images ===
print("Downloading images…")
for _, row in tqdm(df.iterrows(), total=len(df)):
    url = row.get(URL_COL)
    bid = row.get(ID_COL)
    if pd.isna(url) or pd.isna(bid):
        continue
    url = url.strip()
    if not url.lower().startswith(('http://','https://')):
        continue
    ext = os.path.splitext(url)[1].split('?')[0] or '.jpg'
    fn  = f"{int(bid)}{ext}"
    out = os.path.join(IMAGE_DIR, fn)
    if os.path.exists(out):
        continue
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        with open(out, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    except Exception as e:
        print(f"⚠️ Failed {url}: {e}")

# === 3) Feature extractor ===
print("Loading MobileNetV2…")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x   = image.img_to_array(img)
    x   = preprocess_input(np.expand_dims(x,axis=0))
    feats = model.predict(x)
    return feats[0].astype('float32')

# === 4) Build FAISS index ===
print("Extracting features & building FAISS index…")
features, ids = [], []
for fn in tqdm(os.listdir(IMAGE_DIR)):
    try:
        bid = int(os.path.splitext(fn)[0])
    except ValueError:
        continue
    feats = extract_features(os.path.join(IMAGE_DIR, fn))
    features.append(feats)
    ids.append(bid)

features = np.stack(features)
d = features.shape[1]
index = faiss.IndexFlatL2(d)
index.add(features)
print(f"Indexed {index.ntotal} images.")

# Save index & ids
faiss.write_index(index, INDEX_PATH)
with open(IDS_PATH, 'wb') as f:
    pickle.dump(ids, f)

# === 5) Evaluate overall accuracy ===
print("Evaluating overall accuracy…")
correct_top1 = 0
correct_topk = 0
results = []

for fn in tqdm(os.listdir(IMAGE_DIR)):
    try:
        true_id = int(os.path.splitext(fn)[0])
    except:
        continue

    qf = extract_features(os.path.join(IMAGE_DIR, fn)).reshape(1,-1)
    D, I = index.search(qf, TOP_K)
    preds = [ids[i] for i in I[0]]

    hit1 = (preds[0] == true_id)
    hitk = (true_id in preds)
    correct_top1 += int(hit1)
    correct_topk += int(hitk)

    meta = df[df[ID_COL]==true_id].iloc[0].to_dict()
    results.append({
        'filename': fn,
        'true_id': true_id,
        'pred_top1': preds[0],
        'hit_top1': hit1,
        f'preds_top{TOP_K}': preds,
        f'hit_top{TOP_K}': hitk,
        'distance_top1': float(D[0][0]),
        **{k: meta[k] for k in meta if k not in (ID_COL,URL_COL)}
    })

total = len(results)
top1_acc = correct_top1/total if total else 0
topk_acc = correct_topk/total if total else 0

print(f"Evaluated {total} images")
print(f"Top-1 Accuracy: {correct_top1}/{total} = {top1_acc:.2%}")
print(f"Top-{TOP_K} Accuracy: {correct_topk}/{total} = {topk_acc:.2%}")

# Save detailed results
res_df = pd.DataFrame(results)
res_df.to_csv(EVAL_OUT_CSV, index=False)
