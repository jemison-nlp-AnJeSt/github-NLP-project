import pandas as pd
import numpy as np

import nltk

#visualizations:
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import GridspecLayout
from wordcloud import WordCloud


#hypothesis testing
import scipy.stats as stats



def top_common_languages(master_df):
    '''
    This function takes in the master_df dataframe and displays the total count and percentage of the Top 10 most used languages in the 
    Linux repositories we sourced. Additionally it plots a chart of the percentages next to the dataframe.
    '''
    master_language_count=pd.concat([master_df.language.value_counts(), master_df.language.value_counts(normalize=True)], axis = 1). head(10)
    master_language_count.columns = ['total_count', 'percentage']
    #plotting out the percentages of the Top 10 languages used in all repos:
    # plt.figure(figsize=(10,8))
    # sns.barplot(data=master_language_count, x = master_language_count.index, y = 'percentage')
    # plt.title('Percentages of the Top 10 Linux Repo languages')

    out_box1 = widgets.Output(layout={"border":"1px solid black"})
    out_box2 = widgets.Output(layout={"border":"1px solid black"})

    with out_box1:
        display(master_language_count)

    with out_box2:
        fig, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(x = master_language_count.index, y = master_language_count.percentage)
        plt.title(f"Percentages of the Top 10 Linux Repo languages")
        plt.ylabel('Percent')
        plt.show()

    grid = GridspecLayout(10, 4)
    grid[:, 0] = out_box1
    grid[:, 1:4] = out_box2

    return grid

def readme_avg_length(master_df):
    '''
    This function takes in the master_df dataframe and creates a plot to displaying the average readme length separated by language.
    '''
    plt.figure(figsize=(18, 12))
    sns.barplot(data = master_df.groupby('language').mean().reset_index().sort_values('length_of_readme', ascending=False), x = 'length_of_readme', y='language', palette="crest")
    plt.title('Average Readme length by Language')
    plt.show()

def linux_corpus(master_df):
    ''' 
    This function takes in the master_df and creates a corpus of all words for all repositories contained in a pandas Series.
    '''
    corpus_words = ' '.join(master_df.clean_readme.apply(lambda r: ' '.join(r)))
    corpus_words = corpus_words.split()
    return pd.Series(corpus_words)

def top_20_linux_bigrams(linux_corpus):
    ''' 
    This function takes in the master_df and creates a visual for the top 20 bigrams in the all the repos.
    '''
    #top bigrams of Shell language in Linux repos:
    top_20_corpus_bigrams = (pd.Series(nltk.ngrams(linux_corpus, 2)).value_counts().head(20))
    top_20_corpus_bigrams.sort_values().plot.barh(color='#1E8FBF', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Corpus bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_corpus_bigrams.reset_index().sort_index(ascending=False)['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def top_20_linux_trigrams(linux_corpus):
    ''' 
    This function takes in the master_df and creates a visual for the top 20 trigrams in the all the repos.
    '''
    #top bigrams of Shell language in Linux repos:
    top_20_corpus_trigrams = (pd.Series(nltk.ngrams(linux_corpus, 3)).value_counts().head(20))
    top_20_corpus_trigrams.sort_values().plot.barh(color='#1E8FBF', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Corpus trigrams')
    plt.ylabel('Trigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_corpus_trigrams.reset_index().sort_index(ascending=False)['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def linux_corpus__trigram_wordcloud(linux_corpus):
    ''' 
    This function takes in the master_df and creates a wordcloud visual for the top 20 trigrams in the all the repos.
    '''
    top_100_linux_trigrams = (pd.Series(nltk.ngrams(linux_corpus, 3)).value_counts().head(100))
    data = {k[0] + ' ' + k[1]: v for k, v in top_100_linux_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def shell_corpus(master_df):
    ''' 
    This function takes in the master_df and creates a corpus of all words for the shell language contained in a pandas Series.
    '''
    shell_words = master_df[master_df.language == 'Shell'].clean_readme.reset_index()
    shell_corpus = []
    for i in range(len(shell_words)):
        shell_corpus.extend(shell_words.clean_readme[i])
    return pd.Series(shell_corpus)

def top_20_shell_bigrams(shell_corpus):
    ''' 
    This function takes in the master_df and creates a visual for the top 20 bigrams for repos using the shell language most.
    '''
    #top bigrams of Shell language in Linux repos:
    top_20_shell_bigrams = (pd.Series(nltk.ngrams(shell_corpus, 2)).value_counts().head(20))
    top_20_shell_bigrams.sort_values().plot.barh(color='#1E8FBF', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Shell bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_shell_bigrams.reset_index().sort_index(ascending=False)['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def python_corpus(master_df):
    ''' 
    This function takes in the master_df and creates a corpus of all words for the python language contained in a pandas Series.
    '''
    python_words = master_df[master_df.language == 'Python'].clean_readme.reset_index()
    python_corpus = []
    for i in range(len(python_words)):
        python_corpus.extend(python_words.clean_readme[i])
    return pd.Series(python_corpus)

def top_20_python_bigrams(python_corpus):
    ''' 
    This function takes in the master_df and creates a visual for the top 20 bigrams for repos using the python language most.
    '''
    #top bigrams of Python language in Linux repos:
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_corpus, 2)).value_counts().head(20))
    top_20_python_bigrams.sort_values().plot.barh(color='#1E8FBF', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Python bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_python_bigrams.reset_index().sort_index(ascending=False)['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def C_corpus(master_df):
    ''' 
    This function takes in the master_df and creates a corpus of all words for the C language returns results in a pandas Series.
    '''
    C_words = master_df[master_df.language == 'C'].clean_readme.reset_index()
    C_corpus = []
    for i in range(len(C_words)):
        C_corpus.extend(C_words.clean_readme[i])
    return pd.Series(C_corpus)

def top_20_C_bigrams(C_corpus):
    ''' 
    This function takes in the master_df and creates a visual for the top 20 bigrams for repos using the C language most.
    '''
    #top bigrams of C language in Linux repos:
    top_20_C_bigrams = (pd.Series(nltk.ngrams(C_corpus, 2)).value_counts().head(20))
    top_20_C_bigrams.sort_values().plot.barh(color='#1E8FBF', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring C bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_C_bigrams.reset_index().sort_index(ascending=False)['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def arch_subset():
    '''
    This function uses the arch_linux_data json file to get a subset of data for only arch linux distributions. It returns out the 
    data as a Pandas DataFrame object.
    '''
    #calling in df
    arch_df = pd.read_json('arch_linux_data.json')
    #adding on cleaned/normalized data column:
    arch_df['cleaned_readme'] = arch_df.readme_contents.apply(c.clean_data)
    #adding on cleaned repo length
    arch_df['cleaned_length'] = 0
    for i in range(len(arch_df.cleaned_readme)):
        arch_df['cleaned_length'][i] = len(arch_df.cleaned_readme[i])
    arch_df.head()
    return arch_df.dropna().reset_index()

def arch_langs(arch_df):
    '''
    This function pulls in arch_df and visulizes the frequency of languages appearing in the repos for arch linux.
    '''
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    arch_df.language.value_counts().head(20).sort_values().plot.barh()
    plt.title('Most common repo languages for Archlinux')
    plt.ylabel('Language')
    plt.xlabel('Count of Occurances')

def arch_corpus(arch_df):
    '''
    This function pulls in arch_df and returns out a Pandas Series containing the words of the arch linux corpus.
    '''
    arch_corpus_list = []
    language = []
    for entry in range(len(arch_df.readme_contents)):
        language.append(arch_df.language[entry])
        arch_corpus_list.extend(c.clean_data(arch_df.readme_contents[entry]))
    arch_corpus = pd.Series(arch_corpus_list)
    return arch_corpus


