from transformers import pipeline
import gradio as gr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
nltk.download('vader_lexicon')
from deep_translator import (GoogleTranslator)
from langdetect import detect

zero_shot_classifier = pipeline("zero-shot-classification" , model='roberta-large-mnli')

spam_detector = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

issues = ["Misconduct" , "Negligence" , "Discrimination" , "Corruption" , "Violation of Rights" , "Inefficiency" ,
           "Unprofessional Conduct", "Response Time" , "Use of Firearms" , "Property Damage"]

apprecn = ["Tech-Savvy Staff" , "Co-operative Staff" , "Well-Maintained Premises" , "Responsive Staff"]

def translate(input_text):
    source_lang = detect(input_text)
    translated = GoogleTranslator(source=source_lang, target='en').translate(text=input_text)
    return translated
    
def spam_detection(input_text):
    
    return spam_detector(input_text)[0]['label'] == 'clean'

def sentiment_analysis(input_text):
    
    score = SentimentIntensityAnalyzer().polarity_scores(input_text)


    del score['compound']

    label = list(filter(lambda x: score[x] == max(score.values()), score))[0]


    
    
    if label == 'neg':
        
        return ["Negative Feedback" , score['neg']]
    
    elif label == 'pos':
        
        return ["Positive Feedback" , -1]
    
    else:
        
        return ["Neutral Feedback" , -1]

def positive_zero_shot(input_text):
    
    return zero_shot_classifier(input_text , candidate_labels = apprecn , multi_label = False)['labels'][0]
    
    
def negative_zero_shot(input_text):
    
    return zero_shot_classifier(input_text , candidate_labels = issues , multi_label = False)['labels'][0]
    
def pipeline(input_text):

    input_text = translate(input_text)
    
    if spam_detection(input_text):
        
        if sentiment_analysis(input_text)[0] == "Positive Feedback":
            
            return "Positive Feedback" , -1 , positive_zero_shot(input_text)
        
        elif sentiment_analysis(input_text)[0] == "Negative Feedback":
            
            return "Negative Feedback" , sentiment_analysis(input_text)[1] ,  negative_zero_shot(input_text)
        
        else:
            
            return "Neutral Feedback" , -1 , ""
    else:
        return "Spam" , ""

iface = gr.Interface(fn = pipeline , inputs=['text'] , outputs=['text' , 'text' , 'text'])
iface.launch(share=True)



