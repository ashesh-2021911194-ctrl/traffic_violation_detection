# Traffic Violation Detection

A deep learning project for detecting traffic violations using computer vision. This system can identify violations such as helmet usage among bike riders.

## Features

- **Object Detection**: Detects bikes, riders, and helmet states
- **YOLO-based Model**: Uses YOLOv8 architecture for real-time detection
- **Web Interface**: Flask-based web application for easy visualization
- **Model Evaluation**: Comprehensive metrics and inference speed benchmarking

## Project Structure

```
├── app.py                 # Flask web application
├── detect.py              # Detection inference script
├── evaluate_model.py      # Model evaluation and benchmarking
├── requirements.txt       # Python dependencies
├── model/                 # Pre-trained models
│   ├── best_fixed.pt      # PyTorch model
│   └── best_fixed.onnx    # ONNX model
├── templates/             # HTML templates
│   └── index.html         # Web UI
└── static/                # Static files (CSS, JS)
```

## Classes Detected

1. **Bike** - Motorcycle/bike detection
2. **Helmet_Off** - Rider without helmet
3. **Helmet_On** - Rider with helmet
4. **No_helmet** - No helmet visible
5. **Rider** - Motorcycle rider

## Model Performance

- **Precision**: 93.34%
- **Recall**: 93.35%
- **mAP@50**: 96.51%
- **Inference Speed**: ~110ms per image (9.0 FPS)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd traffic_violation_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application
```bash
python app.py
```
Then navigate to `http://localhost:5000` in your browser.

### Command Line Detection
```bash
python detect.py --source <image_or_video_path>
```

### Model Evaluation
```bash
python evaluate_model.py
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- YOLOv8 (ultralytics)

See `requirements.txt` for complete dependencies.

## License

MIT License

## Author

Your Name

## Contact

For questions or issues, please open an issue on GitHub.
