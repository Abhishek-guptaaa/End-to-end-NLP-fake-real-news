from flask import Blueprint, request, render_template, jsonify
from .pipeline import run_pipeline

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        prediction = run_pipeline(text)
        return jsonify({'prediction': prediction})
    return render_template('index.html')
