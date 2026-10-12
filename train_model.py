import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

x=[]
y=[]
with open('SMSSpamCollection', 'r') as file:
    reader = csv.reader(file,delimiter="\t")
    for row in reader:
        x.append(row[1])
        y.append(row[0])

vectorizer=CountVectorizer()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=67,stratify=y)
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

model=MultinomialNB()
model.fit(X_train_vec,y_train)
predictions=model.predict(X_test_vec)

# accuracy = accuracy_score(y_test, predictions)
# precision = precision_score(y_test, predictions, pos_label="spam")
# recall = recall_score(y_test, predictions, pos_label="spam")
# f1Score = f1_score(y_test, predictions, pos_label="spam")

joblib.dump(vectorizer,"spam_vectorizer.pkl")
joblib.dump(model,"spam_model.pkl")