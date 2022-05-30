import pandas as pd
from utils import pickle_dump, process_text
import nltk

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


if __name__ == "__main__":
    print("Loading...")
    df = pd.read_csv(r"labeled_data/labeled_products_final_clean.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    nltk.download("stopwords")

    print("Vectorizing...")
    messages = CountVectorizer(analyzer=process_text)
    messages_bow = messages.fit_transform(df["text"])
    X_train, X_test, y_train, y_test = train_test_split(
        messages_bow, df["illegal"], test_size=0.20, random_state=0
    )

    print("Creating and training the Naive Bayes...")
    classifier = MultinomialNB().fit(X_train, y_train)

    pickle_dump(classifier, "finalized_model.sav")
    pickle_dump(messages, "count_vector")

    print("Evaluating Naive Bayes on training data set...")
    print(classifier.predict(X_train))
    print(y_train.values)
    pred = classifier.predict(X_train)
    print(classification_report(y_train, pred))
    print("Confusion Matrix: \n", confusion_matrix(y_train, pred))
    print()
    print("Accuracy: \n", accuracy_score(y_train, pred))

    print("Evaluating Naive Bayes on test data set...")
    print(classifier.predict(X_test))
    print(y_test.values)
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, pred))
    print()
    print("Accuracy: \n", accuracy_score(y_test, pred))

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, pred),
        display_labels=classifier.classes_
    )
    disp.plot()
    plt.show()