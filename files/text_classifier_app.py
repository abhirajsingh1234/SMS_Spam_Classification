import tkinter as tk
from tkinter import ttk, scrolledtext
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classifier")
        self.root.geometry("800x600")
        
        # Initialize NLTK components
        self.ps = PorterStemmer()
        
        # Load the models
        try:
            self.vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
            self.model = pickle.load(open('model.pkl', 'rb'))
        except FileNotFoundError:
            print("Error: Model files not found!")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input area
        input_label = ttk.Label(main_frame, text="Enter text to classify:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.text_area = scrolledtext.ScrolledText(main_frame, width=70, height=10)
        self.text_area.grid(row=1, column=0, pady=5, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        classify_btn = ttk.Button(button_frame, text="Classify", command=self.classify_text)
        classify_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_text)
        clear_btn.grid(row=0, column=1, padx=5)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Classification Result", padding="5")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.result_label = ttk.Label(output_frame, text="")
        self.result_label.grid(row=0, column=0, pady=5)
        
        # Preprocessed text display
        preprocess_frame = ttk.LabelFrame(main_frame, text="Preprocessed Text", padding="5")
        preprocess_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.preprocess_label = ttk.Label(preprocess_frame, text="", wraplength=700)
        self.preprocess_label.grid(row=0, column=0, pady=5)
    
    def transform_text(self, x):
        x = x.lower()
        x = nltk.word_tokenize(x)
        y = []
        for i in x:
            if i.isalnum():
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(self.ps.stem(i))
        return ' '.join(y)
    
    def classify_text(self):
        # Get input text
        input_text = self.text_area.get("1.0", tk.END).strip()
        
        if input_text:
            try:
                # Preprocess the text
                preprocessed_text = self.transform_text(input_text)
                self.preprocess_label.config(text=f"Preprocessed: {preprocessed_text}")
                
                # Vectorize the text
                vectorized_text = self.vectorizer.transform([preprocessed_text])
                
                # Make prediction
                prediction = self.model.predict(vectorized_text)
                
                if prediction[0] == 0:
                    self.result_label.config(text=f"Classification Result: {'non-spam'}")
                else:
                    self.result_label.config(text=f"Classification Result: {'spam'}")


                # Display result
                
                
            except Exception as e:
                self.result_label.config(text=f"Error: {str(e)}")
        else:
            self.result_label.config(text="Please enter SMS")
    
    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        self.result_label.config(text="")
        self.preprocess_label.config(text="")

def main():
    root = tk.Tk()
    app = TextClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 