from flask import Flask, request, render_template, jsonify
import os
import webbrowser
from threading import Timer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_name = "Gnider/new_3ep_512_roberta_normal"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

encoder_path = "label_encoder.joblib"
label_encoder = joblib.load(encoder_path)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and file.filename.endswith('.txt') and file.content_length <= 5242880:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': True, 'filename': filename})
    else:
        return jsonify({'success': False, 'error': 'Invalid file format or size'})

@app.route('/upload/<filename>')
def upload_success(filename):
    return render_template('upload.html', filename=filename)

@app.route('/classify/<filename>')
def classify(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='cp1251') as f:
                text = f.read()
        except UnicodeDecodeError:
            return jsonify({'result': 'Error: Unable to read the file with both UTF-8 and cp1251 encodings.'})

    # Placeholder for your custom genre classification function
    result = your_genre_classification_function(text)

    return jsonify({'result': result})

def chunk_text(text, tokenizer, max_length=512, stride=128):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < max_length:
            chunk = torch.cat((chunk, torch.zeros(max_length - len(chunk), dtype=torch.long)))
        chunks.append(chunk)
    
    return torch.stack(chunks)
    
def your_genre_classification_function(text, max_length=512, stride=128):
    chunks = chunk_text(text, tokenizer, max_length, stride)
    all_logits = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = {'input_ids': chunk.unsqueeze(0), 'attention_mask': (chunk != 0).unsqueeze(0)}
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits.squeeze().cpu().numpy().copy())

    all_logits = np.array(all_logits)
    mean_logits = all_logits.mean(axis=0)

    probabilities = torch.nn.functional.softmax(torch.tensor(mean_logits), dim=-1)

    top_3_prob, top_3_catid = torch.topk(probabilities, 3)
    top_3_prob = top_3_prob.squeeze().numpy().tolist()
    top_3_catid = top_3_catid.squeeze().numpy().tolist()

    top_3_labels = label_encoder.inverse_transform(top_3_catid)

    result = []
    for label, prob in zip(top_3_labels, top_3_prob):
        result.append(f"{label}: {prob * 100:.2f}%")

    return "\n".join(result)
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False)
