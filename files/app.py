from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

# Load the models
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found!")

# Initialize NLTK components
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))
    return ' '.join(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the SMS text from form
        sms_text = request.form['sms_text']
        
        # Preprocess the text
        transformed_text = transform_text(sms_text)
        
        # Vectorize
        vector_input = vectorizer.transform([transformed_text])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Store in session
        session['result'] = int(result)
        session['sms_text'] = sms_text
        
        # Redirect to GET request
        return redirect(url_for('index'))
    
    # For GET requests
    result = session.pop('result', None)  # Get and remove from session
    sms_text = session.pop('sms_text', None)  # Get and remove from session
    
    return render_template('index.html', result=result, sms_text=sms_text)

if __name__ == "__main__":
    app.run(debug=True)


