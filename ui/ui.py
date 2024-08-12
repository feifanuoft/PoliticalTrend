import gradio as gr
import torchtext
import torch
import matplotlib.pyplot as plt
import args_parameter
import seaborn as sns

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification,GPT2Tokenizer, GPT2ForSequenceClassification
import numpy 
import CNN_model
import torch.nn.functional as F
args = args_parameter.args
glove = torchtext.vocab.GloVe(name="6B", dim=100)


def ui(text1,text2, model):
    a = ""
    b = ""
    prob1 = 0
    prob2 = 0
    if model == "CNN":
        cnn_model1 = CNN_model.CNNTextClassifier(glove,args)
        cnn_model1.load_state_dict(torch.load('./model_parameters/cnn1.pt'))

        tokens = text1.split()
        token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1)
        cnn_pre = cnn_model1(token_tensor)
        cnn_prob = float(torch.sigmoid(cnn_pre))
        if cnn_prob > 0.5:
            a,prob1 = "progressive", cnn_prob
        else:
            a,prob1 = "conservative", -(1-cnn_prob)

        cnn_model2 = CNN_model.CNNTextClassifier(glove,args)
        cnn_model2.load_state_dict(torch.load('./model_parameters/cnn2.pt'))

        tokens = text2.split()
        token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1)
        cnn_pre = cnn_model2(token_tensor)
        cnn_prob = float(torch.sigmoid(cnn_pre))
        if cnn_prob > 0.5:
            b,prob2 = "progressive", cnn_prob
        else:
            b,prob2 = "conservative", -(1-cnn_prob)

    if model == "BERT":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bertmodel1 = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
        bertmodel1.load_weights('./model_parameters/bert1.h5')
        input_text = text1

        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="tf")
        predictions = bertmodel1(input_ids)


        logits = predictions.logits
        probabilities = tf.nn.softmax(logits, axis=1)
        prob1 = tf.reduce_max(probabilities)
        prob1 =  float(prob1.numpy())
        label = tf.argmax(probabilities[0])
        if label.numpy() == 0:
            a = "conservative"
            prob1 = -abs(prob1)
        else:
            a = "progressive"
        
        bertmodel2 = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
        bertmodel2.load_weights('./model_parameters/bert2.h5')
        input_text = text2

        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="tf")
        predictions = bertmodel1(input_ids)


        logits = predictions.logits
        probabilities = tf.nn.softmax(logits, axis=1)
        prob2 = tf.reduce_max(probabilities)
        prob2 =  float(prob2.numpy())
        label = tf.argmax(probabilities[0])
        if label.numpy() == 0:
            b = "liberalism"
            prob2 = -abs(prob2)
        else:
            b = "regulationism"
        


    if model == "GPT2":
        gptmodel1 = GPT2ForSequenceClassification.from_pretrained("./model_parameters/gpt2_1")

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        inputs = tokenizer(text1, return_tensors='pt')

        outputs = gptmodel1(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        softmax_logits = F.softmax(logits[0], dim=0)
        max_value, _ = torch.max(softmax_logits, dim=0)
        label_mapping = {0: 'conservative', 1: 'progressive'}
        predicted_class = label_mapping[predicted_labels.item()]
        a = predicted_class
        if a == "conservative":
            prob1 = -max_value.item()
        if a == "progressive":
            prob1 = max_value.item()
        
        gptmodel2 = GPT2ForSequenceClassification.from_pretrained("./model_parameters/gpt2_2")

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        inputs = tokenizer(text2, return_tensors='pt')

        outputs = gptmodel2(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        softmax_logits = F.softmax(logits[0], dim=0)
        max_value, _ = torch.max(softmax_logits, dim=0)
        label_mapping = {0: 'liberalism', 1: 'regulationism'}
        predicted_class = label_mapping[predicted_labels.item()]
        b = predicted_class
        if b == "liberalism":
            prob2 = -max_value.item()
        if b == "regulationism":
            prob2 = max_value.item()
            


    fig, ax = plt.subplots()

    sns.set_style("whitegrid")

    # Scatter plot
    ax.scatter(prob1, prob2, marker='o', s=400, edgecolors="red", facecolors="green")


    ax.set_title("Political Leaning by " + model)
    ax.set_xlabel('Liberalism')
    ax.set_ylabel('Conservative', color='b')

    ax2 = ax.twiny()

    
    ax2.set_xlabel('Regulationism', color='r')



    ax3 = ax.twinx()


    ax3.set_ylabel('Progressive', color='g')

    ax.axhline(0, color='black', linewidth=2, linestyle='-')
    ax.axvline(0, color='black', linewidth=2, linestyle='-')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    return a,b,prob1,prob2,fig

inputs = [
        "text",
        "text",
        gr.Dropdown(['CNN', 'BERT', 'GPT2'], label="model")
    ]

box1 = gr.Textbox(label="Model prediction(social)")
box2 = gr.Textbox(label="Model prediction(economic)")
box3 = gr.Textbox(label="Prediction probability(social)")
box4 = gr.Textbox(label="Prediction probability(economic)")

figs = gr.Plot()

demo = gr.Interface(fn=ui, inputs=inputs, outputs=[box1,box2,box3,box4,figs])

demo.launch(debug=True)