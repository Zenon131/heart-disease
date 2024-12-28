# Heart Disease Prediction Model

This project implements a machine learning model to predict the presence of heart disease using various medical attributes. The model uses a Random Forest Classifier to make predictions based on patient data.

## Features

- Data visualization using correlation matrix
- Random Forest Classifier implementation
- Model performance evaluation using confusion matrix and classification report
- Prediction labeling (No Heart Disease/Possible Heart Disease)

## Dataset

The model uses a dataset containing the following features:
- age: Age of the patient
- sex: Gender of the patient
- trestbps: Resting blood pressure
- chol: Serum cholesterol level
- fbs: Fasting blood sugar
- restecg: Resting electrocardiographic results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels colored by fluoroscopy
- thal: Thalassemia type
- target: Presence of heart disease (0 = No, 1 = Yes)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Ensure you have the required dataset (`heart.csv`) in the correct location
2. Run the script:
```bash
python main.py
```

The script will:
1. Load and preprocess the data
2. Display a correlation matrix visualization
3. Train a Random Forest Classifier
4. Make predictions on the test set
5. Display classification metrics and a confusion matrix

## Model Details

- Algorithm: Random Forest Classifier
- Number of estimators: 1000
- Feature selection: sqrt
- Test set size: 60% of data
- Random state: 42

## Output

The program outputs:
- Individual predictions for each sample
- A detailed classification report showing precision, recall, and F1-score
- A confusion matrix visualization

## License

This project is open source and available under the MIT License.
