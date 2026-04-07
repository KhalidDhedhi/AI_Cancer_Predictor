# AI Cancer Predictor

A machine learning application that predicts whether a cell nuclei sample is malignant or benign based on various cellular measurements using Logistic Regression.

## Project Overview

This project implements a cancer prediction system using a Logistic Regression model trained on the Breast Cancer Wisconsin dataset. The application provides both a command-line interface for model training and a web-based interface for making predictions.

## Project Structure

```
AI_Cancer_Predictor/
├── Application/
│   └── app.py              # Streamlit web application for predictions
├── Asset/
│   └── style.css           # CSS styling for the web interface
├── Data/
│   └── data.csv            # Breast Cancer Wisconsin dataset
├── Model/
│   ├── main.py             # Model training script
│   ├── model.pkl           # Trained logistic regression model
│   └── scaler.pkl          # Fitted StandardScaler for feature normalization
├── .vscode/
│   └── settings.json       # VS Code settings
└── README.md               # Project documentation
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- streamlit
- joblib
- plotly
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KhalidDhedhi/AI_Cancer_Predictor.git
cd AI_Cancer_Predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the machine learning model:

```bash
cd Model
python main.py
```

This will:
- Load the cancer dataset
- Split data into training and testing sets (80/20 split)
- Train a Logistic Regression model
- Display accuracy and classification report
- Save the trained model and scaler for later use

### Running the Web Application

To launch the interactive Streamlit web application:

```bash
streamlit run Application/app.py
```

The application will open in your default web browser. You can then:
- Use sliders to input cell nuclei measurements
- Get a real-time prediction of whether the sample is benign or malignant
- View visualization of the predictions

### Model Features

The model uses 30 features based on cell nuclei measurements:

**Features for Mean, Standard Error, and Worst:**
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

### Model Performance

The Logistic Regression model achieves high accuracy on the test set. Performance metrics are displayed when training the model.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains:
- 569 samples
- 30 features (cell nuclei measurements)
- 2 classes: Malignant (M) and Benign (B)

## Technologies Used

- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Preprocessing**: StandardScaler

## File Descriptions

- **Application/app.py**: Main Streamlit application with interactive UI for predictions
- **Model/main.py**: Model training script that creates and evaluates the ML model
- **Model/model.pkl**: Serialized trained Logistic Regression model
- **Model/scaler.pkl**: Serialized StandardScaler for feature normalization
- **Data/data.csv**: Breast Cancer Wisconsin dataset
- **Asset/style.css**: Custom CSS styling

## How the Prediction Works

1. User inputs cell nuclei measurements through the Streamlit interface
2. The input data is normalized using the fitted StandardScaler
3. The trained Logistic Regression model predicts the probability
4. Results are displayed with a visualization indicating benign or malignant classification

## Future Enhancements

- Implement additional classification models (SVM, Random Forest, Neural Networks)
- Add model comparison and evaluation metrics
- Implement cross-validation for better model evaluation
- Add data visualization and exploratory data analysis
- Deploy the application to a cloud platform

## Contributing

Feel free to fork the project and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please contact Khalid Dhedhi.

## Dataset Citation

Breast Cancer Wisconsin (Diagnostic) Data Set from UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic