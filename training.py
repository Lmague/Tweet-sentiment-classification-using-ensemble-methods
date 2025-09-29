import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
import matplotlib.pyplot as plt
import numpy as np


def load_train(path="Data/training_data.csv"):
    df = pd.read_csv(path)
    y = df["Sentiment"].astype(int) # On ne garde que la colonne des labels et on la convertit en int au cas où si ce n'est pas déjà le cas
    X = df.drop(columns=["Sentiment"]) # On garde toutes les colonnes avec les features
    return X, y

def load_test(path="Data/test_data.csv"):
    df = pd.read_csv(path)
    NonFeatures = df[["ItemID", "SentimentText"]]
    X_testData = df.drop(columns=["ItemID", "SentimentText"])
    return NonFeatures, X_testData

def scores(y_true, y_pred): # On fait une fonction avec tout les scores pour les calculer en une fois pour tout les modèles
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    X, y = load_train("Data/training_data.csv")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y ) # On split les données en train et validation (80% train, 20% validation) et on met un random Stat de 42 pour que le résultat soit le même à chaque fois. On met stratify = y pour que la répartition des classes soit la même dans les deux sets (train et val) (Pas forcément nécéssaires avec autant de tweet et seulemnt 2 classes mais mieux vaut faire ça au cas où)
    models = { # On crée un dictionnaire avec les modèles qu'on veut tester
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators = 200, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators = 200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, random_state=4, max_depth=2),
        "Bagging": BaggingClassifier(n_estimators = 200, random_state=42),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train) # On entraine le modèle
        y_pred = model.predict(X_val) # On fait des prédictions sur les données de validation
        results[model_name] = scores(y_val, y_pred) # On calcule les scores et on les stocke dans le dictionnaire results

    model_names = list(results.keys())
    acc  = [results[m]["accuracy"] for m in model_names]
    prec = [results[m]["precision"] for m in model_names]
    rec  = [results[m]["recall"] for m in model_names]
    f1   = [results[m]["f1"] for m in model_names]

    def barplot(title, values):
        plt.figure()
        x = np.arange(len(model_names))
        plt.bar(x, values)
        plt.xticks(x, model_names, rotation=20)
        plt.ylim(0, 1)
        plt.ylabel("score")
        plt.title(title)
        for xi, v in zip(x, values):
            plt.text(xi, min(v + 0.02, 1.0), f"{v:.3f}", ha="center", va="bottom")
        plt.tight_layout()

    barplot("Accuracy par modèle", acc)
    barplot("Precision par modèle", prec)
    barplot("Recall par modèle", rec)
    barplot("F1-score par modèle", f1)


    # Cette partie sert à sauvegarder les figures dans un dossier images mais les ayant déja pré-générées, je ne fais pas tourner le code à chaque fois. C'est juste pour montrer le processus.    
    '''for fig, name in zip(map(plt.figure, plt.get_fignums()), 
                        ["accuracy.png","precision.png","recall.png","f1.png"]):
        fig.savefig("images/" + name, dpi=200, bbox_inches="tight")'''

    # Pour choisir le meilleur modèle, nous utilisnons le score F1 car il utilise le precision score et le recall
    best_model_name = max(results, key=lambda model: results[model]["f1"])
    best_model = models[best_model_name]
    print("\n=== Résultats F1 des différents modèles ===")
    for model_name, metrics in results.items():
        print(f"{model_name:<20} : {metrics['f1']:.3f}")
    print("===========================================")

    print("\n======= Résultats du meilleur modèle ======")
    print(f"Modèle choisi : {best_model_name}\n")
    for metric, value in results[best_model_name].items():
        print(f"{metric:<10} : {value:.3f}")
    print("==========================================\n")

    best_model.fit(X, y) # On réentraine le meilleur modèle sur toutes les données d'entrainement (train + val) pour avoir un modèle plus robuste

    NonFeatures, X_testData = load_test("Data/test_data.csv") # On charge les données de test

    y_test_pred = best_model.predict(X_testData) # On fait des prédictions sur les données de test avec le meilleur modèle (réentrainé sur toutes les données train)

    NonFeatures["PredictedSentiment"] = y_test_pred # On ajoute les prédictions dans le dataframe NonFeatures pour avoir un dataframe avec ItemID, SentimentText et PredictedSentiment

    NonFeatures.to_csv("test_predictions.csv", index=False)
    print("Prédictions sauvegardées dans 'test_predictions.csv'")


if __name__ == "__main__" :
    print("Entrainement des modèles...")
    main()