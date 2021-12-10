from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import cross_origin
import lyrics_classifier
import json

app = Flask(__name__)

@app.route('/get_genre', methods=['GET','POST'])
@cross_origin()
def get_genre():
    data = request.data
    data = str(data, 'utf-8')
    json_data = json.loads(data)
    result = lyrics_classifier.predict_genre([json_data['lyrics']['lyrics']])
    return jsonify({'result': result})

if __name__ == "__main__":
    app.debug = True
    app.run()
