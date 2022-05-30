import heapq
import pandas as pd
from utils import pickle_load, process_text, word_count

if __name__ == "__main__":
    print("Loading data, frequent words and classifier...")
    df = pd.read_csv(r"data/test_data.csv")
    classifier = pickle_load("finalized_model.sav")
    count_vector = pickle_load("count_vector")

    print("Preprocessing...")
    df["text"] = df["text"].apply(process_text, args=(True,))

    print("Classifying...")
    result = classifier.predict(count_vector.transform(df["text"]))

    print("Saving...")
    df["illegal"] = pd.DataFrame(result)
    df.to_csv("out.csv", index=False)

    N = 15
    print(f"Calculating TOP {N} most frequent irregular words (eCatalog)...")
    df_e_irregular = pd.read_csv(
        r"./labeled_data/labeled_irregular_data_ecatalog.csv"
    )
    df_e_irregular["text"] = df_e_irregular["text"].apply(process_text)
    count = word_count(df_e_irregular["text"])
    most_freq_irregular = heapq.nlargest(N, count, key=count.get)
    d_e = {
        "text": most_freq_irregular,
        "frequency": [count[key] for key in most_freq_irregular]
    }
    df_most_irregular = pd.DataFrame(d_e, columns=["text", "frequency"])
    file_name = f"top_{N}_most_freq_irregular"
    df_most_irregular.to_csv(rf"./reports/{file_name}.csv", index=False)

    print(f"Calculating TOP {N} most frequent regular words (mercado livre)... ")
    df_ml = pd.read_csv(r"./labeled_data/labeled_regular_data_ml.csv")
    df_ml["text"] = df_ml["text"].apply(process_text)
    count = word_count(df_ml["text"])
    most_freq_ml = heapq.nlargest(N, count, key=count.get)
    d_ml = {"text": most_freq_ml, "frequency": [count[key] for key in most_freq_ml]}
    df_most_regular_ml = pd.DataFrame(d_ml, columns=["text", "frequency"])
    file_name = f"top_{N}_most_freq_ml"
    df_most_regular_ml.to_csv(rf"./reports/{file_name}.csv", index=False)


