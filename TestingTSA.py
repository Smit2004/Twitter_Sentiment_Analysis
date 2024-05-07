import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib
from tqdm import tqdm


# Define the UI elements
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x300")

upload_label = tk.Label(root, text="Upload a CSV file:")
upload_label.pack()

upload_button = tk.Button(root, text="Upload", command=lambda: upload_csv())
upload_button.pack()

label = tk.Label(root, text="Enter text:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

output_label = tk.Label(root, text="")
output_label.pack()

# Define functions
def upload_csv():
    filename = filedialog.askopenfilename()
    data = pd.read_csv(filename)
    analyze_data(data)

def analyze_sentiment():
    model = joblib.load('model.pkl')
    new_text = entry.get()
    new_features = vectorizer.transform([new_text])
    predicted_sentiment = model.predict(new_features)[0]
    output_label.config(text="Predicted sentiment: " + str(predicted_sentiment))

def analyze_data(data):
    # Preprocess the text data using TfidfVectorizer
    global vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    features = vectorizer.fit_transform(data['SentimentText'])

    # Define the models to use
    knn = KNeighborsClassifier()
    nb = MultinomialNB()
    svm = LinearSVC()

    # Perform 10-fold cross-validation to determine the best model
    knn_scores = cross_val_score(knn, features, data['Sentiment'], cv=10)
    nb_scores = cross_val_score(nb, features, data['Sentiment'], cv=10)
    svm_scores = cross_val_score(svm, features, data['Sentiment'], cv=10)

    # Calculate the mean accuracy of each model
    knn_accuracy = knn_scores.mean()
    nb_accuracy = nb_scores.mean()
    svm_accuracy = svm_scores.mean()

    # Print the accuracies of all the models
    print("KNN accuracy: ", knn_accuracy)
    print("Naive Bayes accuracy: ", nb_accuracy)
    print("SVM accuracy: ", svm_accuracy)

    # Select the best model based on cross-validation
    if svm_accuracy > max(knn_accuracy, nb_accuracy):
        best_model = svm
        saved_model = "SVM"
    elif nb_accuracy > max(knn_accuracy, svm_accuracy):
        best_model = nb
        saved_model = "Naive Bayes"
    else:
        best_model = knn
        saved_model = "KNN"

    # Train the selected model on the entire dataset
    best_model.fit(features, data['Sentiment'])

    # Save the best model
    joblib.dump(best_model, 'model.pkl')

    # Print which model is saved
    print("Best model saved: ", saved_model)

# Add a button to trigger the sentiment analysis
button = tk.Button(root, text="Analyze", command=analyze_sentiment)
button.pack()

# Start the UI loop
root.mainloop()