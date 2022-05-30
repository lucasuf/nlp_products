import pandas as pd
from utils import process_text


def mercado_livre():
    """
    Faz leitura e atribui rotulo aos dados obtidos na raspagem feita no mercadolivre.com
    """
    df = pd.read_json(r"./data/ml.json")
    df["illegal"] = 0
    df.to_csv(
        r"./labeled_data/labeled_regular_data_ml.csv",
        index=None,
        columns=["title", "illegal"],
        header=["text", "illegal"],
    )
    return df


def e_catalog_regulars():
    """
    Faz leitura e atribui rotulo aos dados regulares da eCatalog
    """
    df = pd.read_csv(r"./data/eCatalog_regulars.csv")
    df["text"] = (df["name"] + " " + df["description"] + " " + df["category__name"] + " " + df["category__description"])

    df["illegal"] = 0
    df.to_csv(
        r"./labeled_data/labeled_regular_data_ecatalog.csv",
        index=None,
        columns=["text", "illegal"],
    )
    return df


def e_catalog_irregulars():
    """
    Faz leitura e atribui rotulo aos dados irregulares da eCatalog
    """
    df = pd.read_csv(r"./data/eCatalog_irregulars.csv")
    df["text"] = (
            df["name"] + " " +
            df["description"] + " " +
            df["category__name"] + " " +
            df["category__description"]
    )

    df["illegal"] = 0
    df.to_csv(
        r"./labeled_data/labeled_irregular_data_ecatalog.csv",
        index=None,
        columns=["text", "illegal"],
    )
    return df


def generate_labeled_data():
    mercado_livre()
    e_catalog_regulars()
    e_catalog_irregulars()


if __name__ == "__main__":
    generate_labeled_data()

    df_ml = pd.read_csv(
        r"./labeled_data/labeled_regular_data_ml.csv"
    )
    df_e_regular = pd.read_csv(
        r"./labeled_data/labeled_regular_data_ecatalog.csv"
    )
    df_e_irregular = pd.read_csv(
        r"./labeled_data/labeled_irregular_data_ecatalog.csv"
    )

    df_final = pd.concat([df_ml, df_e_regular, df_e_irregular])
    df_final["illegal"] = df_final["illegal"].astype("int")
    df_final.to_csv(
        r"./labeled_data/labeled_products_final.csv",
        index=False,
        columns=["text", "illegal"],
    )

    df_final["text"] = df_final["text"].apply(process_text, args=(True,))
    df_final.to_csv(
        r"./labeled_data/labeled_products_final_clean.csv",
        index=False,
        columns=["text", "illegal"],
    )

