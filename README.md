# Analyse de sentiments de tweet avec l'aide de différents algorithmes.

Afin de mieux lire ce fichier si vous n'êtes pas sur github, je conseille de l'ouvrir dans un editeur qui supporte les fichier .md comme obsidian ou VSCode (Ctrl+Shift+V sur Windows/Linux et Cmd+Shift+V sur MacOS)

## 1 - Présentation du projet, des algorithmes et des scores utilisés


Pour ce projet, nous devions analyser les sentiments de tweets en trouvant des features et les faire passer dans différents algorithmes.  
Les algorithmes utilisés sont :  
- Un arbre de décision (Decision Tree)  
- Une forêt aléatoire (Random Forest)  
- AdaBoost  
- Gradient Boosting  
- Bagging  

L'objectif du projet étant de trouver quel modèle est le plus efficace pour l'analyse de tweet avec nos configurations (les résultats peuvent varier en fonctions des hyperparamètres, des features créées voir même de la seed aléatoire utilisée)

Pour détecter quel est l'algorithme le plus efficace sur nos données, nous avons utilisé 4 scores différents qui mesurent la performance des prédictions du modèle. Ces 4 scores sont :  

- **Accuracy**  
	- L’accuracy est un score qui se calcule de la manière suivante :  
	$$Acc = \frac{\text{Nombre de données bien classifiées}}{\text{Nombre de données totales}}$$  

- **Précision**  
	- La précision mesure le nombre de vrais positifs parmi les positifs classifiés. Elle se calcule de cette manière :  
	$$Precision = \frac{\text{Nombre de vrais positifs}}{\text{Nombre de vrais + faux positifs}}$$  

- **Recall**  
	- Le recall mesure le nombre de données positives correctement classifiées. Il se calcule de cette manière :  
	$$Recall = \frac{\text{Nombre de vrais positifs}}{\text{Nombre de vrais positifs + Nombre de faux négatifs}}$$  

- **F1**  
	- Le score F1 allie précision et rappel et se calcule de la manière suivante :  
	$$F1 = 2\times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}$$  

Nous utiseront le score F1 de manière génerale pour comparer les modèles

## 2 - Préparation des données

### 2.1 Charger les données

Pour charger les dataset, nous utilisont la métode ```python read_csv(file_path, encoding =)``` avec un try and catch pour essayer 2 types d'encodage différents car les émojis peuvent de temps en temps poser problème en utf-8 et latin1 peut combler ses lacunes.

### 2.2 Nettoyer et préparer les données

Avant toutes opérations sur les données, nous nous assuront que celles-ci sont bien toutes des chaînes de caractères avec cette ligne ```python df = df["SentimentText"].astype(str)```. Cette opéartion permet d'éviter des erreurs de types plus tard.

Afin d'éviter tout problèmes lors de la création de features et plus tard pour l'entrainement des modèles, nous avons décider de verifier si il n'y avait pas de données manquantes dans les datasets et si les données ne sont pas vides (tweet inexistant, catégorie absente pour le dataset de train ou juste des espaces en guise de donnée). Pour faire ca, nous faisont de cette manière : 
```python
    df = df.drop_duplicates()
    df = df.dropna(subset=["SentimentText"]) # Supprimer les lignes où le texte est manquant ou vide après nettoyage
    df = df[df["SentimentText"].str.strip() != ""] # Ici, on enlève les lignes où le texte est juste des espaces et on enleve les espaces avant et après le texte
    
    if is_train and "Sentiment" in df.columns: # Supprimer les lignes sans label dans le train
        df = df.dropna(subset=["Sentiment"])
    return df
```


### 2.3 Créer les features

Les données étaient initialement de simples tweets ce qui est insuffisant comme données pour analyser leurs sentimetns car les modèles de machine learning utilisé ne peuvent pas analyser du texte brut. Pour résoudre ce problème, nous avons extrait 12 features qui nous permettrait d'extraire ce qui, selon nous, semble être les informations les plus importantes à utiliser.

Les 12 features sont les suivantes :
- **Le nombre de mots**
- **Les émojis joyeux**
- **Les émojis triste**
- **Les mots positifs**
- **Les mots négatifs**
- **Les élongations de mots**
- **Les majuscules**
- **Le nombre de point d'exclamtion (!)**
- **Le nombre de point d'interogation (?)**
- **Le nombre de point de suspension (...)**
- **Le nombre de mentions (@)**
- **Le nombre d'hashtag (#)**



Pour les features comme les émojis ou les émotions des mots, nous avons utilisé des dictionnaires non exhaustif afin d'avoir une manière de les détecter. Ces méthodes ne sont pas infaillibles mais elles restent suffisement interessantes.

### 2.4 Finalisation

À la suite de ces étapes, nous supprimons l'ID et le texte des données de training pour avoir un dataset exclusivement consacré à l'entrainement des modèles. Nous utilisons ces lignes pour le faire :
```python 
if not test :
    df = df.drop(columns=["SentimentText", "ItemID"])
```

Nous nous assuront tout de même à ne pas le faire pour les données de test car nous voulons garder les tweets originaux pour analyser manuellement les résultats.

Nous enregistrons ensuite ces nouveaux dataset en .csv avec cette fonction ```python save_data(df, output_path)``` qui créé un nouveau fichier .csv avec le dataframe **df** à la position **output_path**.
