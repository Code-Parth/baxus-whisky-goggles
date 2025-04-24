# Whisky Goggles

A computer vision system that scans whisky bottle labels and matches them to a database of 501 bottles, allowing users to quickly identify bottles and record pricing information at liquor stores.

## System Architecture

![System-Architecture](https://github.com/user-attachments/assets/bd9c0b25-f80a-4c64-99e6-7c5e7e3eec26)


## Overview

Whisky Goggles uses deep learning and similarity search to recognize whisky bottles from images with high accuracy. The system extracts visual features from bottle images using MobileNetV2 and performs efficient matching using FAISS.

## Features

- **Upload or Camera**: Choose between uploading an image or using your device's camera
- **Bottle Recognition**: Instantly identify bottles with a single click
- **Visual Results**: See confidence scores visualized as bar charts
- **Detailed Information**: View comprehensive bottle details in a table format
- **Multiple Models**: Switch between different trained models
- **Download Results**: Save recognition results for later reference

## Installation

```bash
# Clone the repository
git clone https://github.com/Code-Parth/baxus-whisky-goggles.git
cd baxus-whisky-goggles
```


```bash
# Install dependencies
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
python prepare_data.py
```

```bash
streamlit run app.py
```

Or run directly in Google Colab using the provided notebook. 
<p><a href="https://colab.research.google.com/github/Code-Parth/baxus-whisky-goggles/blob/master/Baxus-Whisky-Goggles.ipynb" target="_parent"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/></a></p>

## How It Works

1. **Feature Extraction**: The system uses MobileNetV2, a convolutional neural network pre-trained on ImageNet, to extract 1280-dimensional feature vectors from each bottle image.

2. **Similarity Search**: FAISS (Facebook AI Similarity Search) creates an index for efficient similarity search among feature vectors.

3. **Matching**: When a new image is provided, the system:
   - Extracts its feature vector
   - Computes similarity with all indexed bottle images
   - Returns the closest matches

4. **Confidence Score**: The system calculates a confidence score for each match based on the distance between feature vectors.

### Component Architecture

![Component-Architecture](https://github.com/user-attachments/assets/98734022-c29f-4185-a164-2495b6fa88e2)


### Bottle Identification Process

![Bottle-Identification-Process](https://github.com/user-attachments/assets/9eaaf1f8-64d7-4ee0-9e11-5c6b225d825f)

## Performance

The system's performance is measured using two metrics:

- **Top-1 Accuracy**: Percentage of cases where the correct bottle is the top match
- **Top-K Accuracy**: Percentage of cases where the correct bottle is among the top K matches

### Accuracy Results

Our system achieved impressive accuracy on the 501-bottle dataset:

| Metric | Accuracy |
|--------|----------|
| Top-1 Accuracy | 0.92 |
| Top-5 Accuracy | 0.97 |

This means:
- 92% of the time, the system correctly identifies the exact bottle as the top match
- 97% of the time, the correct bottle appears within the top 5 matches

These results demonstrate the system's high reliability in real-world whisky identification scenarios.

## Demo Video



https://github.com/user-attachments/assets/335bbc19-aa3e-4cdb-93f1-3b2a65e1b294



## Limitations

- Performance may vary depending on image quality, lighting, and angle
- The system works best with clear shots of bottle labels
- Current model is optimized for a dataset of 501 bottles

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MobileNetV2 pre-trained model from TensorFlow/Keras
- FAISS library from Facebook Research
