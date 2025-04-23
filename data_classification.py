import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json
import faiss
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


class BottleTrainingCLI:
    def __init__(self):
        self.id_col = 'id'
        self.url_col = 'image_url'
        self.name_col = 'name'
        self.top_k = 5
        self.model_name = 'baxus_white_bottle_model'
        self.image_dir = 'bottle_images'
        self.model_dir = 'model'

        # Model data
        self.bottle_df = None
        self.model = None
        self.index = None
        self.features = []
        self.ids = []
        self.metadata = []

        # Create directories if they don't exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def load_dataset(self, csv_path):
        """Load bottle dataset from CSV file"""
        print(f"Loading dataset from {csv_path}...")

        try:
            self.bottle_df = pd.read_csv(csv_path)
            print(f"Successfully loaded {len(self.bottle_df)} bottle records.")

            # Display sample data
            print("\nSample data:")
            print(self.bottle_df.head())

            # Display column information
            print("\nColumn information:")
            for col in self.bottle_df.columns:
                print(f"- {col}")

            # Check if required columns exist
            missing_columns = []
            if self.id_col not in self.bottle_df.columns:
                missing_columns.append(self.id_col)
            if self.url_col not in self.bottle_df.columns:
                missing_columns.append(self.url_col)

            if missing_columns:
                print(
                    f"\nWarning: Required columns not found: {', '.join(missing_columns)}")
                print(
                    f"Expected columns: {self.id_col} (ID), {self.url_col} (URL)")
                print(
                    "Please update the CSV or modify the column mapping in the script.")
                return False

            return True
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return False

    def download_images(self):
        """Download bottle images from URLs in the dataset"""
        if self.bottle_df is None:
            print("Error: Please load a dataset first.")
            return False

        print(
            f"Downloading images using ID column '{self.id_col}' and URL column '{self.url_col}'...")

        total = len(self.bottle_df)
        downloaded = 0

        # Use tqdm for progress bar
        for _, row in tqdm(self.bottle_df.iterrows(), total=total, desc="Downloading"):
            url = row.get(self.url_col)
            bid = row.get(self.id_col)

            # Skip if URL or ID is missing
            if pd.isna(url) or pd.isna(bid):
                continue

            # Make sure it's a string
            url = str(url).strip()
            if not url.lower().startswith(('http://', 'https://')):
                continue

            # Build a safe filename
            ext = os.path.splitext(url)[1].split('?')[0] or '.jpg'
            filename = f"{int(bid)}{ext}"
            out_path = os.path.join(self.image_dir, filename)

            # Download if not already present
            if not os.path.exists(out_path):
                try:
                    r = requests.get(url, stream=True, timeout=5)
                    r.raise_for_status()
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                    downloaded += 1
                except Exception as e:
                    print(f"\nWarning: Failed to download {url}: {e}")
            else:
                downloaded += 1

        print(
            f"\nFinished downloading {downloaded} images to {self.image_dir}")
        return True

    def extract_features(self, img_path):
        """Extract features from an image using MobileNetV2"""
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(np.expand_dims(x, axis=0))
        feats = self.model.predict(x, verbose=0)
        return feats[0].astype('float32')

    def build_index(self):
        """Build FAISS index from bottle images"""
        if self.bottle_df is None:
            print("Error: Please load a dataset first.")
            return False

        if not os.path.exists(self.image_dir) or not os.listdir(self.image_dir):
            print(
                f"Error: No images found in {self.image_dir}. Please download images first.")
            return False

        print("Loading feature extraction model (MobileNetV2)...")
        self.model = MobileNetV2(
            weights='imagenet', include_top=False, pooling='avg')

        print("Building FAISS index...")

        # Extract all features
        self.features = []
        self.ids = []
        self.metadata = []

        # Get list of image files
        image_files = os.listdir(self.image_dir)
        total = len(self.bottle_df)
        processed = 0

        for _, row in tqdm(self.bottle_df.iterrows(), total=total, desc="Processing"):
            bid = row[self.id_col]
            # Find downloaded file
            matches = [f for f in image_files if f.startswith(str(bid))]
            if not matches:
                continue

            # Extract features
            try:
                img_path = os.path.join(self.image_dir, matches[0])
                feats = self.extract_features(img_path)
                self.features.append(feats)
                self.ids.append(bid)

                # Collect metadata (all columns except id)
                meta = {k: row[k] for k in row.index if k != self.id_col}
                self.metadata.append(meta)

                processed += 1
            except Exception as e:
                print(
                    f"\nWarning: Error extracting features from {matches[0]}: {e}")

        # Create and populate FAISS index
        if len(self.features) > 0:
            features_array = np.stack(self.features)
            d = features_array.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(features_array)

            print(f"\nSuccessfully indexed {self.index.ntotal} bottles")
            return True
        else:
            print("Error: No features could be extracted from images.")
            return False

    def evaluate_accuracy(self):
        """Evaluate accuracy of the model"""
        if self.bottle_df is None or self.index is None:
            print("Error: Please load a dataset and build index first.")
            return False

        print(f"Evaluating model accuracy (Top-{self.top_k})...")

        # Settings
        results = []
        correct_top1 = 0
        correct_topk = 0
        total = 0

        # Get image files
        image_files = os.listdir(self.image_dir)

        # Loop through every downloaded image
        for fname in tqdm(image_files, desc="Evaluating"):
            # Only consider files named like "123.jpg"
            try:
                true_id = int(os.path.splitext(fname)[0])
            except ValueError:
                continue

            img_path = os.path.join(self.image_dir, fname)

            # Extract features & search
            try:
                qf = self.extract_features(img_path).reshape(1, -1)
                D, I = self.index.search(qf, self.top_k)
                preds = [self.ids[i] for i in I[0]]

                hit1 = (preds[0] == true_id)
                hitk = (true_id in preds)

                total += 1
                correct_top1 += int(hit1)
                correct_topk += int(hitk)
            except Exception as e:
                print(f"\nWarning: Error evaluating {fname}: {e}")

        # Compute accuracies
        top1_acc = correct_top1 / total if total else 0
        topk_acc = correct_topk / total if total else 0

        # Store results
        self.eval_results = {
            'top1_acc': top1_acc,
            'topk_acc': topk_acc,
            'top_k': self.top_k,
            'total': total,
            'correct_top1': correct_top1,
            'correct_topk': correct_topk
        }

        # Display results
        print("\nEvaluation Results:")
        print(f"Total images: {total}")
        print(f"Top-1 Correct: {correct_top1}")
        print(f"Top-{self.top_k} Correct: {correct_topk}")
        print(f"Top-1 Accuracy: {top1_acc:.2%}")
        print(f"Top-{self.top_k} Accuracy: {topk_acc:.2%}")

        # Optionally create and save a chart
        try:
            self._save_accuracy_chart()
        except Exception as e:
            print(f"Warning: Could not create accuracy chart: {e}")

        return True

    def _save_accuracy_chart(self):
        """Create and save accuracy charts to files"""
        if not hasattr(self, 'eval_results'):
            return

        results = self.eval_results

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart of overall accuracies
        metrics = {
            'Top-1 Accuracy': results['top1_acc'],
            f'Top-{results["top_k"]} Accuracy': results['topk_acc']
        }
        ax1.bar(metrics.keys(), metrics.values())
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overall Identification Accuracy')

        # Hit vs Miss counts for Top-1
        hit_count = results['correct_top1']
        miss_count = results['total'] - hit_count
        ax2.bar(['Hit Top-1', 'Miss Top-1'], [hit_count, miss_count])
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Top-1 Hit vs. Miss Counts')

        # Save to file
        plt.tight_layout()
        plt.savefig('accuracy_chart.png')
        print("Saved accuracy chart to 'accuracy_chart.png'")

    def export_model(self):
        """Export the trained model for use in inference app"""
        if self.index is None:
            print("Error: Please build the index first.")
            return False

        try:
            # Create model directory
            model_dir = os.path.join(self.model_dir, self.model_name)
            os.makedirs(model_dir, exist_ok=True)

            print(f"Exporting model to {model_dir}...")

            # Save FAISS index
            index_path = os.path.join(model_dir, "index.faiss")
            faiss.write_index(self.index, index_path)
            print(f"- Saved FAISS index to {index_path}")

            # Save IDs and metadata
            ids_path = os.path.join(model_dir, "ids.pickle")
            with open(ids_path, 'wb') as f:
                pickle.dump(self.ids, f)
            print(f"- Saved bottle IDs to {ids_path}")

            metadata_path = os.path.join(model_dir, "metadata.pickle")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"- Saved metadata to {metadata_path}")

            # Save configuration
            config = {
                'model_name': self.model_name,
                'feature_extractor': 'MobileNetV2',
                'id_column': self.id_col,
                'name_column': self.name_col,
                'created_date': pd.Timestamp.now().isoformat(),
                'feature_dimension': self.index.d,
                'num_bottles': self.index.ntotal
            }

            if hasattr(self, 'eval_results'):
                config['evaluation'] = {
                    'top1_accuracy': self.eval_results['top1_acc'],
                    'topk_accuracy': self.eval_results['topk_acc'],
                    'top_k': self.eval_results['top_k']
                }

            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"- Saved configuration to {config_path}")

            print(f"\nModel successfully exported to: {model_dir}")
            return True

        except Exception as e:
            print(f"Error: Failed to export model: {e}")
            return False

    def run_pipeline(self, csv_path):
        """Run the complete pipeline: load → download → build → evaluate → export"""
        print("\n=== STEP 1: Loading Dataset ===")
        if not self.load_dataset(csv_path):
            return False

        print("\n=== STEP 2: Downloading Images ===")
        if not self.download_images():
            return False

        print("\n=== STEP 3: Building Index ===")
        if not self.build_index():
            return False

        print("\n=== STEP 4: Evaluating Accuracy ===")
        if not self.evaluate_accuracy():
            return False

        print("\n=== STEP 5: Exporting Model ===")
        if not self.export_model():
            return False

        print("\n=== Pipeline Completed Successfully ===")
        print(
            f"Model exported to: {os.path.join(self.model_dir, self.model_name)}")
        print("You can now use this model with the Streamlit inference app.")
        return True


def parse_args():
    """Parse command line arguments - only CSV path is required"""
    parser = argparse.ArgumentParser(
        description='Bottle Recognition Training Tool')
    parser.add_argument(
        'csv', type=str, help='Path to CSV file with bottle data')
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create trainer instance
    trainer = BottleTrainingCLI()

    # Run the full pipeline
    trainer.run_pipeline(args.csv)


if __name__ == "__main__":
    main()
