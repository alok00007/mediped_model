from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from chat import get_response

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/medipedChat')
db = client['MedipedChatbot']
collection = db['Chatbot']

@app.route('/', methods=['GET'])
def index_get():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.get_json().get("message")
    response = get_response(text)

    message = {"answer": response}
    # Store chat data in MongoDB
    chat_data = {
    'user_message': text,
    'bot_response': response
    }
    collection.insert_one(chat_data)
    
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
