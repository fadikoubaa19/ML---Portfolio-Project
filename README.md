# Jarvis is a chatbot builded for website named DevApp
![image](https://user-images.githubusercontent.com/66315303/168298027-86040365-d7a5-4bcc-a794-f9a1e59c83e8.png)



## Installation:
### Create a virtual enviremment:
````bash
 $ python3 -m venv venv
 ````
 
 ### Activate the envirement:
 ````bash
 $ . venv/bin/activate
````

### Install Pytorch & NLTK:
```bash
$ python
>>> import nltk
>>> nltk.download('punkt')
```
### Files:
#### Front-end part containes 3 files:
* basic.html: The structure and context of Jarvis
* bot.css: Design and decoration file of the chat bot
* app.js



 #### Intents.json:
*  is a json file that contains the responses of Jarvis, it's easy to modify or delete it, you only need to Add ("tag","responses","patterns"):
```bash
   {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": [
        "See you later, thanks for visiting",
        "Have a nice day",
        "Bye! Come back again soon."
      ]
    },
    {
      "tag": "projects",
      "patterns": [
        "when can i do the live projects",
        "When do i do the live projects ?"
      ],
      "responses": [
        "The live projects will be available at the end of each course",
        "At the conclusion of each course, live projects will be available. "
      ]
    },
 ```    
  
  #### App.py:
 * After making sure that the virtual envirement is succesfully 
    activated and the flask is installed, we render the template 
    to generate the output that's based from Jinja2 & initialize the 
    Flask-Cors extension with default arguments in order to allow 
    CORS for all domains on all routes & predict the flask api.
 
```bash
    from flask import Flask, render_template
```
* Enable debug mode using Flask cors (Cross Origin Resource Sharing):
```bash
from flask_cors import CORS
    app.run(debug=True)
```
* Decorate the flask route
```bash
@app.get("/")
def index_get():
    return render_template("base.html")
 ```
 
 #### nltk_utils.py:
 * In this file we used Nltk & numpy:
 * so at the first we need to split sentences into array of words/tokens
 * Then we returned the tokenized sentence:
 ```bash
 nltk.word_tokenize(sentence)
 ```
 * In the next function we needed to find the root form of the words in lowercase For ex:
 words = ["take", "takes", "taking"]
 words = [stem(w) for w in words]
 -> [take, take, take]
 ```bash
 stemmer.stem(word.lower())
 ```
 * In the end we returned the bag of words(answers) lets an example:
 sentence = ["hello", "how", "are", "you"]
 the word = hello
 --> [1,0,0,0]
 * 1 for each word exist in the sentence otherwise 0.
 * here i filled the word with zeros using numpy & set the data type into float32:
 ```bash
 bag = np.zeros(len(words), dtype=np.float32)
 ```
 * I returned 1 for each word exist by her index using enumrate and return the current word and index:
 ```bash
     for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
 ```
#### train.py:
![image](https://user-images.githubusercontent.com/66315303/168298443-74e6e3f2-5619-450f-8908-4059f541ae26.png)

* First of all we need to collect all words
* ignore punctuation and caractere
* loop all the paterns
* Calling tokenize,stem & ignored wordes and put it in the same list(all_words).
* stem & lower each word:
```bash
all_words = [stem(w) for w in all_words if w not in ignore_words]
```
* Sort words using sort (we need to filtre only the unique words):
```bash
all_words = sorted(set(all_words))
```
* Sort tags for unique labels:
```bash
tags = sorted(set(tags))
```
* Create a bag of words so we created 2 empty lists:
```bash
x_train=[]
y_train=[]
```
* create a loop that contains the patterns and sentences to give each pattern_sentence a bag of words
& then we append it:
```bash
 bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
  ```
  * Put in the label for patterns
  * Using pytorche we computes the difference between two probability distributions for provided set:
  ```bash
   label = tags.index(tag)
    y_train.append(label)
  ```
  * Convert x_train & y_train to arrays so that's why we need to import numpy:
  ```bash
  import numpy as np
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  ```
  * To Create pytorche data set & load it, we need to import torch:
  ```bash
 import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
  ```
  * create a new class named Chatdataset contain 3 different function (__init__,__getitem__,__len__):
  ```bash
  class ChatDataset(Dataset):
```
 * The Init function is created to store the data in x_train & y_train:
 ```bash
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
 ```
 * function getitem is a function that return the stored data:
 ```bash
     def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
  ```
  * function len call the length of dataset to retrun the size:
  ```bash 
   def __len__(self):
        return self.n_samples
```
* the chatdataset class is create to iterate automatically and for better training data.

* Declare hyperparametre:
* this hyperparametre define the length or the size of each pack of wards created.
```bash
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)
```
* Use gpu or cpu device to load data:
```bash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
#### model.py:
* Create 3 different layers:
```bash
 self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        
````
* Activation function:
```bash
        self.relu = nn.ReLU()
```
#### data.pth:
* we found the trained data in this file
### Libraries:
* Torch:An open source machine learning framework that accelerates the path from research prototyping to production deployment.
* numpy:NumPy is a Python library used for working with arrays. 
* Flask:Flask is a web framework, it's a Python module that lets you develop web applications easily
* flask_cors: Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.
* nltk:The Natural Language Toolkit (NLTK) is a platform used for building Python programs that work with human language data f





