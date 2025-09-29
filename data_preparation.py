import pandas as pd # Pour manipuler les données (les ouvir, les nettoyer, les transformer, etc.)
import re # Pour faire des opérations sur des chaînes de caractères

def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


def clean_data(df): # Pour nettoyer les données (supprimer les doublons et gérer les valeurs manquantes)
    df = df.drop_duplicates()
    if "SentimentText" in df.columns:
        df["SentimentText"] = df["SentimentText"].fillna("")
    return df

def save_data(df, output_path): # Pour sauvegarder les données nettoyées et transformées dans un nouveau fichier CSV
    df.to_csv(output_path, index=False)

# Dictionnaires pour extraction de caractéristiques
happy_emojis = [":)", ";-)", ":D", ":P", ":O", "<3", "XD", "xD", ";-D", ":-)", ";-P", ":-P", ":-O", ":-D", ";-O", ";-D", "^_^",
    "😊", "😉", "👍", "😄", "😃", "😁", "😎", "😇", "🥰", "🤗", "😂", "🤩", "✨", "🌟", "🎉", "❤️", "💕", "😍", "🤣", "😘",
    "🙌", "🥳", "😋", "😌", "😻","👏"]

sad_emojis = [":(", ">:(", ":'(", ":|", ":/", ":*", ":-(", ":-*", ":-/", ":-|", ":-(", ">.<", "😢", "😠", "😞", "👎", "😔", "😩", 
    "😭", "😡", "🤬", "😣", "😤", "💔", "😒", "🙄", "😫", "😨", "😰", "🤢", "🤮", "😵", "☹️", "😓", "😥"]

positive_words = ["amazing", "awesome", "great", "happy", "love", "excellent", "fantastic",
    "good", "nice", "best", "wonderful", "cool", "like", "enjoyed", "perfect",
    "delightful", "incredible", "pleased", "satisfied", "smiling", "glad",
    "beautiful", "brilliant", "outstanding", "superb", "joyful", "friendly",
    "laughing", "amused", "grateful", "enthusiastic", "positive", "supportive",
    "winning", "charming", "relaxed", "peaceful", "yay", "lit", "slay", "goat",
    "dope", "stellar", "vibes", "thrilled", "blessed", "proud", "epic",
    "legendary", "fire", "wholesome", "adorable", "impressive", "fabulous",
    "inspiring", "lovable", "top-notch", "vibrant", "uplifting", "hilarious",
    "majestic", "smooth", "neat", "elegant", "satisfying", "enjoyable",
    "cheerful", "kind", "helpful", "polite"
]

negative_words = ["terrible","awful","bad","hate","worst","horrible","sad","angry","disappointed","boring",
    "dislike","annoying","disgusting","problem","upset","unhappy","poor","crying",
    "lame","gross","useless","frustrated","rude","offended","tired","mean","depressed",
    "hopeless","scared","unpleasant","worried","sick","insulting","aggressive","confused",
    "rash","fail","cringe","bruh","ugh","smh","miserable","broken","awful","dreadful",
    "horrific","regret","disaster","nasty","loser","foolish","dumb","unfair","toxic",
    "lazy","pathetic","offensive","threatening","scandalous","painful","stressful",
    "flawed","useless","buggy","wrong","unreliable",]




def create_dataset(df):
    s = df["SentimentText"].astype(str) # Pour s'assurer que toutes les entrées sont des chaînes de caractères

    df["word_number"] = s.str.split().map(len)
    df["happy_emoji"] = s.map(lambda x: int(any(emo in x for emo in happy_emojis)))
    df["sad_emoji"] = s.map(lambda x: int(any(emo in x for emo in sad_emojis)))
    df["positive_word"] = s.map(lambda x: sum(word in x.lower() for word in positive_words) if isinstance(x, str) else 0)
    df["negative_word"] = s.map(lambda x: sum(word in x.lower() for word in negative_words) if isinstance(x, str) else 0)
    df["elongations"] = s.map(lambda x: len(re.findall(r"([a-zA-Z])\1{2,}", x.lower())) if isinstance(x, str) else 0)
    df["cap"] = s.map(lambda x: sum(1 for c in x if c.isupper()))
    df["nb_exclamations"] = s.str.count('!')
    df["nb_questions"] = s.str.count(r"\?")
    df["nb_ellipsis"] = s.str.count(r"\.\.\.")
    df["nb_mentions"] = s.str.count('@')
    df["nb_hashtags"] = s.str.count('#')
    return df

def process_data(file_path, output_path, test = False):
    df = load_data(file_path)
    df = clean_data(df)
    df = create_dataset(df)
    if not test :
        df = df.drop(columns=["SentimentText", "ItemID"])
    save_data(df, output_path)


if __name__ == "__main__": # Pour que le code s'execute directemnt quand on l'appelle (pas besoin de mettre d'autres paramètres)
    process_data("Data/train.csv", "Data/training_data.csv", test=False)
    process_data("Data/test.csv", "Data/test_data.csv", test=True)
    print("Données prétraitées et sauvegardées dans 'Data/training_data.csv' et 'Data/test_data.csv'")
