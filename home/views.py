from django.shortcuts import render,HttpResponse
from google.protobuf.text_format import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from home.models import Review

# Create your views here.
loaded_model= None
loaded_tokenizer= None

def load(request):
    global loaded_model
    loaded_model= load_model("./model.h5")
    global loaded_tokenizer
    loaded_tokenizer= pickle.load(open("./tokenizer.pkl","rb"))
    #print('file loaded')
    return render(request,"base.html")

#cleaning text
def cleaning_text(text):
    lemma= WordNetLemmatizer()
    review= str(text)
    review= re.sub('[^a-zA-Z]', " ", review)

    review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]

    return " ".join(review)

#tokenizig and padding
def token_pad(data):
    max_word= 48
    tokens= loaded_tokenizer.texts_to_sequences(data)
    tokens_pad = pad_sequences(tokens, maxlen=max_word)
    return tokens_pad

#getting rating and confidance
def get_class(result):
    rating= np.argmax(result[0])
    confidence = round(np.max(result[0])*100)
    return rating, confidence

    

def predict(request):
    if request.method == "POST":

        input_text = request.POST['message']
        real_text = input_text
        input_text = cleaning_text(input_text)
        input_text = [input_text]
        tokens_pad = token_pad(input_text)
        prediction = loaded_model.predict(tokens_pad)
        rating,confidence = get_class(prediction)

        #print("rating {},confidence {}".format(rating,confidence))
        rating= str(rating+1)

        ins= Review(review= real_text,rating=rating)
        ins.save()
        #stars= "*"*rating
        #context={"rating":rating,"stars":stars,"confidence":confidence}
        
        content= Review.objects.all()
        context= {"content":content}

        return render(request,"result.html",context)
        
    else:
        return render(request,"base.html")
