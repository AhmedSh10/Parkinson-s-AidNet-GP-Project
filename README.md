# 🧠 Parkinson's AidNet - AI-Powered Detection System

<div align="center">
  
  ![Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Tool-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
  ![AI](https://img.shields.io/badge/Field-AI%2FML-00D9FF?style=for-the-badge&logo=artificial-intelligence&logoColor=white)
  
  **Graduation Project 2024**
  
</div>

## 📋 Overview

**Parkinson's AidNet** is a comprehensive AI-powered system designed to aid in the detection and monitoring of Parkinson's disease through multiple modalities including keystroke patterns, voice analysis, and medical image processing. This graduation project combines deep learning models with practical healthcare applications to provide accessible diagnostic support for patients and healthcare professionals.

## 🎥 Demo Video

<div align="center">
  
  [![Parkinson's AidNet Demo](https://img.youtube.com/vi/pAPkpdasH8A/0.jpg)](https://youtu.be/pAPkpdasH8A)
  
  **Click to watch the full demonstration**
  
</div>

## 🎯 Project Goals

- **Multi-Modal Detection**: Combine multiple detection methods for comprehensive diagnosis
- **Early Detection**: Identify early signs of Parkinson's disease through various biomarkers
- **Accessibility**: Provide a cost-effective and accessible diagnostic tool
- **High Accuracy**: Achieve reliable results using state-of-the-art AI models
- **Clinical Support**: Assist healthcare professionals in making informed decisions

## ✨ Key Components

### 1️⃣ TappyKeystroke Analysis
- **Typing Pattern Recognition**: Analyze keystroke dynamics and timing
- **Motor Skill Assessment**: Detect subtle changes in typing behavior
- **Machine Learning Models**: KNN and Random Forest classifiers
- **Early Warning System**: Identify potential symptoms before clinical diagnosis

### 2️⃣ Voice Analysis
- **Speech Pattern Recognition**: Analyze voice characteristics and speech patterns
- **Acoustic Features**: Extract frequency, amplitude, and timing features
- **Vocal Tremor Detection**: Identify voice instabilities associated with Parkinson's
- **Real-Time Processing**: Instant voice analysis and feedback

### 3️⃣ Medical Image Processing
- **DenseNet121 Architecture**: Deep learning for medical image classification
- **MRI/CT Scan Analysis**: Process brain imaging data
- **Feature Extraction**: Identify structural changes in brain regions
- **High Precision**: Achieve clinical-grade accuracy in image analysis

### 4️⃣ Combined Approach
- **Multi-Modal Fusion**: Integrate results from all detection methods
- **Ensemble Learning**: Combine predictions for improved accuracy
- **Comprehensive Diagnosis**: Provide holistic assessment of Parkinson's indicators
- **Confidence Scoring**: Calculate overall probability with confidence intervals

## 🛠️ Tech Stack

### Machine Learning & AI
- **TensorFlow/Keras**: Deep learning model development
- **Scikit-learn**: Machine learning algorithms (KNN, Random Forest)
- **DenseNet121**: Pre-trained CNN for image classification
- **Signal Processing**: Audio and keystroke pattern analysis
- **NumPy/Pandas**: Data manipulation and analysis

### Development Environment
- **Jupyter Notebook**: Interactive development and experimentation
- **Python 3.8+**: Core programming language
- **Matplotlib/Seaborn**: Data visualization
- **Librosa**: Audio processing and feature extraction
- **OpenCV**: Image processing utilities

## 📊 Model Performance

### TappyKeystroke Detection
- **Algorithm**: K-Nearest Neighbors (KNN) & Random Forest
- **Features**: Keystroke timing, pressure, hold time, flight time
- **Accuracy**: High precision in detecting motor skill changes

### Voice Analysis
- **Features**: MFCC, pitch, jitter, shimmer, HNR
- **Processing**: Real-time audio signal analysis
- **Indicators**: Voice tremor, speech rate, articulation

### Image Processing (DenseNet121)
- **Architecture**: 121-layer Convolutional Neural Network
- **Input**: Medical imaging (MRI/CT scans)
- **Output**: Classification with confidence scores
- **Transfer Learning**: Fine-tuned on Parkinson's-specific dataset

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
Jupyter Notebook
Scikit-learn
Librosa (for audio processing)
OpenCV
NumPy, Pandas, Matplotlib
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/AhmedSh10/Parkinson-s-AidNet-GP-Project.git
cd Parkinson-s-AidNet-GP-Project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

4. **Open and run the notebooks** for each component

## 📁 Project Structure

```
Parkinson-s-AidNet-GP-Project/
├── notebooks/              # Jupyter notebooks for each modality
│   ├── TappyKeystroke.ipynb
│   ├── VoiceAnalysis.ipynb
│   ├── ImageProcessing.ipynb
│   └── CombinedModel.ipynb
├── models/                 # Trained model files
│   ├── knn_model.pkl
│   ├── rf_model.pkl
│   └── densenet121.h5
├── data/                   # Datasets
│   ├── keystroke/
│   ├── voice/
│   └── images/
├── src/                    # Source code
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── results/                # Experiment results
└── docs/                   # Documentation
```

## 📈 Usage Examples

### Keystroke Analysis
```python
from src.models import KeystrokeAnalyzer

analyzer = KeystrokeAnalyzer()
result = analyzer.analyze_typing_pattern(keystroke_data)
print(f"Parkinson's Probability: {result['probability']:.2%}")
```

### Voice Analysis
```python
from src.models import VoiceAnalyzer

voice_analyzer = VoiceAnalyzer()
features = voice_analyzer.extract_features(audio_file)
prediction = voice_analyzer.predict(features)
```

### Image Processing
```python
from src.models import ImageClassifier

classifier = ImageClassifier(model='densenet121')
result = classifier.classify(brain_scan_image)
print(f"Classification: {result['class']}, Confidence: {result['confidence']:.2%}")
```

## 🏆 Achievements

- ✅ Multi-modal detection system with high accuracy
- ✅ Integration of three different diagnostic approaches
- ✅ Real-time analysis capabilities
- ✅ User-friendly interface for medical professionals
- ✅ Comprehensive documentation and research

## 👥 Team Members

This project was developed by a dedicated team of researchers and developers:

- **Ahmed Sherif** - Lead Developer & ML Engineer
- **Reham Ashraf** - Voice Analysis Specialist
- **Tasneem Selim** - Image Processing Expert
- **Soad Sabry** - Data Scientist
- **Ebthal Karam** - Research & Documentation

## 🎓 Academic Context

Developed as a **Graduation Project (GP)** for the Computer Science program in 2024, representing cutting-edge research in AI-assisted medical diagnosis.

### Research Areas
- Deep Learning in Medical Diagnosis
- Multi-Modal Machine Learning
- Parkinson's Disease Biomarkers
- Clinical Decision Support Systems

## 🔮 Future Enhancements

- **Mobile Application**: Develop cross-platform mobile app
- **Cloud Integration**: Enable cloud-based processing
- **Telemedicine**: Integrate with telehealth platforms
- **Continuous Monitoring**: Long-term patient tracking
- **Additional Biomarkers**: Incorporate gait analysis and facial expression recognition
- **Clinical Trials**: Validate with larger patient populations

## 📚 Documentation

For detailed information about the project methodology, algorithms, and results, please refer to the attached documentation in the `docs/` folder.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/Improvement`)
3. Commit your changes (`git commit -m 'Add some Improvement'`)
4. Push to the branch (`git push origin feature/Improvement`)
5. Open a Pull Request

## 📄 License

© Parkinson's AidNet Team - Developed for educational and research purposes.

## 🙏 Acknowledgments

- Medical advisors and healthcare professionals
- Parkinson's disease patients who participated in research
- Academic supervisors and mentors
- Open-source community for tools and frameworks

## 📞 Contact

For questions, collaborations, or more information:

**Ahmed Sherif (Lead Developer)**
- GitHub: [@AhmedSh10](https://github.com/AhmedSh10)
- LinkedIn: [Ahmed Sherif](https://linkedin.com/in/dev-ahmed-sherif)

## 📚 References

- Research papers on Parkinson's disease detection
- DenseNet architecture documentation
- Signal processing and machine learning literature
- Clinical studies on Parkinson's biomarkers

---

<div align="center">
  
  **⭐ If you find this project valuable, please consider giving it a star!**
  
  **🧠 Empowering healthcare through AI**
  
  *Developed with ❤️ by Parkinson's AidNet Team*
  
</div>
