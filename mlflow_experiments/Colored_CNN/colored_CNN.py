import numpy as np
import cv2
import mlflow
import mlflow.keras
from keras.models import load_model
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('mlflow_experiments\Colored_CNN\colored_CNN.h5')

# Load and preprocess a new image for testing
def preprocess_image(image_path):
    img_array = cv2.imread(image_path)
    if img_array is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    img_array = cv2.resize(img_array, (70, 70))  # Resize to the same size as the training images
    img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict using the model
def predict_image(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        return predicted_class
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def save_confusion_matrix(cm, categories, filepath):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

# Function to save classification report
def save_classification_report(report, filepath):
    with open(filepath, 'w') as f:
        f.write(report)

# Define experiment name
experiment_name = "Lung and Colon Cancer Classification"
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_param("model", "colored_CNN.h5")
    
    root_folder = 'photos'
    categories = {'colon_aca': 0, 'colon_n': 1, 'lung_aca': 2, 'lung_n': 3, 'lung_scc': 4}
    true_labels = []
    predicted_labels = []
    
    # Loop through each category and image
    for category, label in categories.items():
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_path.endswith('.jpeg'):
                    predicted_class = predict_image(img_path)
                    true_labels.append(label)
                    predicted_labels.append(predicted_class)
    
    # Create confusion matrix and classification report
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=list(categories.keys()))
    print(report)
    
    # Create confusion matrix and classification report
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=list(categories.keys()))
    print(report)
    print(cm)
    
    # Save confusion matrix as an image and log it as an artifact
    cm_filepath = "confusion_matrix.png"
    save_confusion_matrix(cm, list(categories.keys()), cm_filepath)
    mlflow.log_artifact(cm_filepath)
    
    report_filepath = "classification_report.txt"
    save_classification_report(report, report_filepath)
    mlflow.log_artifact(report_filepath)
    
    # Calculate and log metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log the model as an artifact
    model_artifact_path = "model/colored_CNN"
    mlflow.keras.log_model(model, artifact_path=model_artifact_path)
    print(f"Model saved in artifacts: {model_artifact_path}")

print('Experiment completed.')

