import joblib

def predict_message(message):
    message_vec=loaded_vectorizer.transform([message])
    prediction=loaded_model.predict(message_vec)[0]

    classes=list(loaded_model.classes_)
    position=classes.index(prediction)

    probability=loaded_model.predict_proba(message_vec)[0]
    return prediction,probability[position]


loaded_vectorizer=joblib.load("spam_vectorizer.pkl")
loaded_model=joblib.load("spam_model.pkl")

message=input("Enter your message:").strip()
if not message:
    print("Please enter a valid message")
else:
    prediction,probability=predict_message(message)
    percentage=round(probability*100,2)
    print(f"Prediction : {prediction}")
    print(f"Probability:{percentage}%")
