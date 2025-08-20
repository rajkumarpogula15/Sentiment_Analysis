import re
import pickle
import nltk
import tkinter as tk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Initialize stemmer
ps = PorterStemmer()

# Stemming function (inline)
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load model and vectorizer
model = pickle.load(open("models/trained_model.sav", "rb"))
vectorizer = pickle.load(open("models/vectorizer.sav", "rb"))

def predict_tweet(tweet):
    # Preprocess
    tweet_stemmed = stemming(tweet)
    # Vectorize
    tweet_vector = vectorizer.transform([tweet_stemmed])
    # Predict
    prediction = model.predict(tweet_vector)[0]

    if prediction == 0:
        return "Negative 😔"
    elif prediction == 1:
        return "Positive 😊"
    elif prediction == 2:
        return "Neutral 😐"
    else:
        return "Irrelevant 🤔"

# GUI
def run_gui():
    def on_submit():
        tweet = entry.get()
        if not tweet.strip():
            result_label.config(text="⚠️ Please enter a tweet!", fg="red")
            return
        result = predict_tweet(tweet)
        result_label.config(text=f"Prediction: {result}", fg="#333")

    root = tk.Tk()
    root.title("Twitter Sentiment Analysis")
    root.geometry("420x220")
    root.configure(bg="#f0f4f7")

    # Label
    label = tk.Label(root, text="Enter a Tweet:", font=("Arial", 12, "bold"), bg="#f0f4f7", fg="#333")
    label.pack(pady=10)

    # Entry box
    entry = tk.Entry(root, width=45, font=("Arial", 11))
    entry.pack(pady=5)

    # Submit button
    button = tk.Button(root, text="Predict Sentiment", command=on_submit, font=("Arial", 11, "bold"),
                       bg="#4CAF50", fg="white", relief="raised", bd=3)
    button.pack(pady=10)

    # Result Label (shows prediction in the same window)
    result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f0f4f7", fg="#333")
    result_label.pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
