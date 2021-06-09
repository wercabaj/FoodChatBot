import numpy as np
import tflearn
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle
from colorama import Fore, Style

nltk.download('punkt')

stemmer = LancasterStemmer()

print("Przetwarzam plik intents.json.....")
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
foods = ['sweet', 'italian', 'fastfood', 'asian', 'polish']
addresses = ['Ul. Slodka 12', 'Ul. Wloska 12', 'Ul. Szybka 12', "Ul. Kwiatowa 34", "Ul. Polska 21/37"]
begginings = ['Moze ', 'A moze ', 'Masz ochote na ', 'Co powiesz na ' , 'Hmm moze ']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenization
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# data model
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8)
model.save('model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)
model.load('./model.tflearn')


def bow(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence):
    results = classify(sentence)
    for i in intents['intents']:
        if i['tag'] == results[0][0]:
            if i['tag'] in foods:
                return random.choice(begginings) + random.choice(i[ 'responses' ]) + "?"
            elif i['tag'] == 'dontknow':
                return response(random.choice(foods))
            else:
                return random.choice(i['responses'])

def tag(inp):
    results = classify(inp)
    for i in intents[ 'intents' ]:
        if i[ 'tag' ] == results[ 0 ][ 0 ]:
            if i['tag'] != 'dontknow':
                return foods[ foods.index(i[ 'tag' ]) ]
            return random.choice(foods)

def address(sentence):
    results = classify(sentence)
    for i in intents['intents']:
        if i[ 'tag' ] == results[0][0]:
            return addresses[foods.index(i['tag'])]

def chat():
    prev_input = ""
    print(Fore.RED + "ZACZNIJ ROZMOWĘ Z NASZYM FOOD CHATBOTEM - WPISZ STOP, BY ZAKONCZYC" + Style.RESET_ALL)
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "stop":
            break
        elif "nie" == inp.lower() or "inne" in inp.lower():
            print(Fore.YELLOW + "Food ChatBot:" + Style.RESET_ALL, response(prev_input))
            prev_input = tag(prev_input)

        elif "gdzie" in inp.lower() or "adres" in inp.lower() or "ok" in inp.lower() or "tak" in inp.lower():
            print(Fore.YELLOW + "Food ChatBot:" + Style.RESET_ALL, "Adres: " + address(prev_input))

        else:
            print(Fore.YELLOW + "Food ChatBot:" + Style.RESET_ALL, response(inp))
            prev_input = inp


# food chatbot start
chat()
