from flask import Flask, request, render_template
import pickle
import re
import os
import PyPDF2

with open('model_and_tfidf.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)

tfidf = loaded_objects['tfidf_vectorizer']
clf = loaded_objects['model']

category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

app = Flask(__name__)

def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', ' ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text

def read_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['resume']
        if file:
            if file.filename.endswith('.pdf'):
                resume_text = read_pdf(file)
            else:
                try:
                    resume_text = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = file.read().decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            category = category_mapping.get(prediction_id, "Unknown")

            return render_template('index.html', prediction=category)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
