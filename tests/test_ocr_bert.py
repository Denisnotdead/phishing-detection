import pandas as pd
from src.text_pipeline.bert_classifier import PhishingBERTClassifier

bert = PhishingBERTClassifier()
bert.load('D:/phishing-detection/models/bert_phishing')

phishing_text = 'Congratulations: You have a money confirmation Bitcoin You have received 16 345,24 in your account (Bitcoin) Confirmation Is Required'
legit_text = 'Hi Mihai, Just a quick note to let you know that our meeting today has been moved to 3:00 PM. Please bring the latest report with you. Thanks, Anna'

df_phishing = pd.DataFrame({'text': [phishing_text]})
df_legit = pd.DataFrame({'text': [legit_text]})

p1 = float(bert.predict_proba(df_phishing)[0][1])
p2 = float(bert.predict_proba(df_legit)[0][1])

label1 = 'PHISHING' if p1 > 0.5 else 'LEGITIMATE'
label2 = 'PHISHING' if p2 > 0.5 else 'LEGITIMATE'

print(f'Phishing image score: {p1:.4f} -> {label1}')
print(f'Legitimate image score: {p2:.4f} -> {label2}')