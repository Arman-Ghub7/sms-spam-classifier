# SMS Spam Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham** (legitimate messages) using Natural Language Processing (NLP) and Multinomial Naive Bayes.

## Features

- Classifies SMS messages as spam or ham.
- Uses CountVectorizer to convert text into numerical features.
- Uses Multinomial Naive Bayes for classification.
- Displays the predicted class and its estimated probability.
- Saves and loads the trained model using Joblib.
- Separates model training from message prediction, so the model doesn't need to be retrained for every prediction.
- Evaluates performance using accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python** — Core programming language
- **Pandas / CSV module** — Dataset handling, depending on the implementation
- **Scikit-learn** — Text vectorization, model training, and evaluation
- **Joblib** — Saving and loading the trained model

## Dataset

This project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

The dataset contains 5,574 SMS messages labeled as:
- `ham` — Legitimate messages
- `spam` — Unwanted or promotional messages

Each message is converted into numerical features before being passed to the classifier.

## How It Works

1. **Data preparation:** Load the dataset and separate messages from their labels.
2. **Train/test split:** Divide the data into training and testing sets.
3. **Text vectorization:** Use CountVectorizer to convert messages into word-count features.
4. **Model training:** Train a Multinomial Naive Bayes classifier on the training data.
5. **Evaluation:** Measure performance on the held-out test set.
6. **Model persistence:** Save the trained vectorizer and classifier using Joblib.
7. **Prediction:** Load the saved objects and classify new SMS messages without retraining.

## Project Structure

```text
SMS-Spam-Classifier/
│
├── train_model.py
├── predict.py
├── SMSSpamCollection
├── spam_vectorizer.pkl
├── spam_model.pkl
└── README.md
```

- `train_model.py` — Loads the dataset, trains the model, evaluates it, and saves the trained objects.
- `predict.py` — Loads the saved objects and predicts whether user-entered messages are spam or ham.
- `SMSSpamCollection` — The SMS dataset.
- `spam_vectorizer.pkl` — The fitted text vectorizer.
- `spam_model.pkl` — The trained Naive Bayes classifier.
- `README.md` — Project documentation.

## Installation

Make sure Python is installed on your system.

Install the required libraries:

```bash
pip install scikit-learn joblib
```

If your implementation uses Pandas, install it as well:

```bash
pip install pandas
```

## Usage

### 1. Train the model

Run the training script:

```bash
python train_model.py
```

This prepares the data, trains and evaluates the classifier, and saves the vectorizer and model.

### 2. Predict SMS messages

Run the prediction script:

```bash
python predict.py
```

Enter an SMS message when prompted. The program returns the predicted class and its estimated probability.

Example:

```text
Enter your message: Hey, are we meeting tomorrow?

Prediction: ham
Probability: [estimated probability]%
```

The actual prediction and probability depend on the message and the trained model.

## Model Evaluation

The CountVectorizer + Multinomial Naive Bayes model achieved the following results on the held-out test set of 1,393 messages:

| Metric | Score |
|---|---:|
| Accuracy | 98.64% |
| Precision (spam) | 96.15% |
| Recall (spam) | 93.58% |
| F1-score (spam) | 94.85% |

These results describe performance on the selected test split and do not guarantee the same performance on unseen real-world messages.

## Experiments

The project also explored TF-IDF vectorization and different classification decision thresholds.

CountVectorizer achieved a higher F1-score than TF-IDF in the experiments conducted. Threshold experiments demonstrated the trade-off between precision and recall.

## Limitations

- The model learns patterns from its training dataset and may misclassify unfamiliar messages.
- Bag-of-words features do not inherently understand sentence meaning, sarcasm, or context.
- Predicted probabilities are model estimates and may not be well calibrated.
- The classifier is a learning project and should not be relied on as a standalone security or phishing detection system.

## Future Improvements

- Build a graphical user interface.
- Explore improved text preprocessing and feature extraction.
- Investigate model errors and compare alternative classifiers.
- Evaluate probability calibration and decision thresholds on separate validation data.
- Experiment with text embeddings and semantic search.

## Learning Outcomes

Through this project, I practised:

- Data preparation and train/test splitting
- Text vectorization and feature extraction
- Supervised machine learning
- Model evaluation and error analysis
- Precision, recall, F1-score, and confusion matrices
- Saving and loading trained models
- Separating training code from inference code

---

**Project:** SMS Spam Classifier  
**Purpose:** Learning and practising machine learning and NLP with Python.