from flask import Flask, render_template, request
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    if request.method == 'POST':
        news_text = request.form['newsText']
        if len(news_text) < 650:
            message = "The news text must contain at least 650 characters."
        else:
            response = requests.post('http://127.0.0.1:8000/predict', json={'text': news_text})
            prediction = response.json()
            return render_template('index.html', prediction=prediction['prediction'], news_text=news_text,
                                   message=message)
    return render_template('index.html', prediction=None, news_text=None, message=message)


if __name__ == '__main__':
    app.run(debug=True)
