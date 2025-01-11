import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=1,
        random_state=1
    ).generate(" ".join(data))

    fig = plt.figure(1, figsize=(10, 6))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

def get_unigram(text_list, top_n=10):
    unigram_counts = Counter()
    for testo in text_list:
        tokens = word_tokenize(testo.lower())
        unigram_counts.update(tokens)

    # Ottieni gli n-gram più frequenti
    most_frequent = unigram_counts.most_common(top_n)

    # Creazione del grafico
    words, counts = zip(*most_frequent)
    plt.figure(figsize=(10, 6))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Frequenza')
    plt.title(f'Unigrams - Top {top_n}')
    plt.gca().invert_yaxis()  # Inverti l'asse y per mostrare la parola più frequente in alto
    plt.show()

    return unigram_counts, most_frequent

def get_bigram(text_list, top_n=10):
    bigram_counts = Counter()
    for testo in text_list:
        tokens = word_tokenize(testo.lower())
        bigram_counts.update(ngrams(tokens, 2))

    # Ottieni gli n-gram più frequenti
    most_frequent = bigram_counts.most_common(top_n)

    # Creazione del grafico
    bigrams, counts = zip(*most_frequent)
    bigrams = [' '.join(bigram) for bigram in bigrams]
    plt.figure(figsize=(10, 6))
    plt.barh(bigrams, counts, color='lightcoral')
    plt.xlabel('Frequenza')
    plt.title(f'Bigrams - Top {top_n}')
    plt.gca().invert_yaxis()  # Inverti l'asse y
    plt.show()

    return bigram_counts, most_frequent

def get_trigram(text_list, top_n=10):
    trigram_counts = Counter()
    for testo in text_list:
        tokens = word_tokenize(testo.lower())
        trigram_counts.update(ngrams(tokens, 3))

    # Ottieni gli n-gram più frequenti
    most_frequent = trigram_counts.most_common(top_n)

    # Creazione del grafico
    trigrams, counts = zip(*most_frequent)
    trigrams = [' '.join(trigram) for trigram in trigrams]
    plt.figure(figsize=(10, 6))
    plt.barh(trigrams, counts, color='lightgreen')
    plt.xlabel('Frequenza')
    plt.title(f'Trigrams - Top {top_n}')
    plt.gca().invert_yaxis()  # Inverti l'asse y
    plt.show()

    return trigram_counts, most_frequent

def plot_distribution(data, plot_col, title=None):
    ax = (
        data.groupby(plot_col)
        .size()
        .plot(
            kind="bar",
            title=title,
            legend=False,
            color=["red", "gray", "green"],
        )
    )
    ax.set_xticklabels(["negative", "neutral", "positive"], rotation=0)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.show()

def plot_text_insights(data, text_column='text'):
    # Calcolare la lunghezza del testo e il numero di parole
    data['text_length'] = data[text_column].apply(len)
    data['word_count'] = data[text_column].apply(lambda x: len(x.split()))

    # Calcolare la lunghezza media e altre statistiche
    avg_text_length = data['text_length'].mean()
    avg_word_count = data['word_count'].mean()
    min_text_length = data['text_length'].min()
    max_text_length = data['text_length'].max()
    min_word_count = data['word_count'].min()
    max_word_count = data['word_count'].max()

    # Creare un dizionario con tutte le informazioni
    insights = {
        'avg_text_length': avg_text_length,
        'avg_word_count': avg_word_count,
        'min_text_length': min_text_length,
        'max_text_length': max_text_length,
        'min_word_count': min_word_count,
        'max_word_count': max_word_count
    }

    # 1. Distribuzione della lunghezza del testo
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], kde=True, color='skyblue')
    plt.title('Distribuzione della Lunghezza del Testo (in caratteri)')
    plt.xlabel('Lunghezza del testo')
    plt.ylabel('Frequenza')
    plt.show()

    # 2. Distribuzione del numero di parole
    plt.figure(figsize=(10, 6))
    sns.histplot(data['word_count'], kde=True, color='lightcoral')
    plt.title('Distribuzione del Numero di Parole per Testo')
    plt.xlabel('Numero di parole')
    plt.ylabel('Frequenza')
    plt.show()

    # 3. Boxplot della lunghezza del testo
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['text_length'], color='lightgreen')
    plt.title('Boxplot della Lunghezza del Testo (in caratteri)')
    plt.xlabel('Lunghezza del testo')
    plt.show()

    # 4. Boxplot del numero di parole
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['word_count'], color='orange')
    plt.title('Boxplot del Numero di Parole per Testo')
    plt.xlabel('Numero di parole')
    plt.show()

    # Restituire le statistiche calcolate
    return insights

def get_stats(input_dataset):

    data = pd.read_csv(input_dataset)

    # Information distribution and relevance
    show_wordcloud(data["text"].values, "Word Cloud")
    plot_distribution(data, "sentiment", "Sentiment Distribution")

    # n-gram analysis
    get_unigram(data["text"].values)
    get_bigram(data["text"].values)
    get_trigram(data["text"].values)

    # other text stats
    plot_text_insights(data)

