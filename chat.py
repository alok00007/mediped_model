import random
import json
import torch
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet

# Load intents data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("data.pth")
model = NeuralNet(data['input_size'], data['hidden_size'], data['output_size']).to(device)
model.load_state_dict(data['model_state'])
model.eval()

# Get response function
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, data['all_words'])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."
