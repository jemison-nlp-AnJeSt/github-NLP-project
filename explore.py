import pandas as pd
import numpy as np
import constants_prepare as c
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

def arch_top_10(arch_corpus):
    '''
    This function takes in arch_corpus and creates a bar graph displaying the top 10 words in the arch corpus.
    '''
    arch_corpus.value_counts().head(10).sort_values().plot.barh()

def top_3_arch_trigrams(arch_corpus):
    '''
    This function takes in arch_corpus and returns a Pandas Series containing the top 3 trigrams in the arch corpus.
    '''
    return (pd.Series(nltk.ngrams(arch_corpus, 3)).value_counts().head(3)) 

def debian_subset():
    '''
    This function uses the debian_data json file to get a subset of data for only debian linux distributions. It returns out the 
    data as a Pandas DataFrame object.
    '''
    df_debian = pd.read_json('debian_data.json')
    df_debian = df_debian[df_debian.language.notnull()]
    return df_debian

def debian_corpus(df_debian):
    '''
    This function takes in the df_debian DataFrame and returns out a complete corpus list for resos in the debian subset.
    '''
    debian_corpus = ' '.join(df_debian['readme_contents'])
    return c.clean_data(debian_corpus)

def debian_unique_words_by_lang(df_debian):
    '''
    This function takes in df_debian and returns a dataframes that shows the unique word information split by language for resos
    in the debian subset.
    '''
    df_debian['clean_readme'] = df_debian.readme_contents.apply(c.clean_data)
    df_debian['total_unique_words'] = df_debian['clean_readme'].apply(lambda r : pd.Series(r).nunique())
    df_debian.groupby('language').total_unique_words.describe().sort_values('count', ascending=False).head(10)

def ubuntu_subset():
    '''This function uses the ubuntu_data json file to get a subset of data for only ubuntu linux distributions. It returns out the 
    data as a Pandas DataFrame object.
     '''
    #calling in df
    ubuntu = pd.read_json('ubuntu_data.json')
    #adding on cleaned/normalized data column:
    ubuntu['cleaned_readme'] = ubuntu.readme_contents.apply(c.clean_data)
    #adding on cleaned repo length
    ubuntu['cleaned_length'] = 0
    for i in range(len(ubuntu.cleaned_readme)):
        ubuntu['cleaned_length'][i] = len(ubuntu.cleaned_readme[i])
    ubuntu.head()
    return ubuntu.dropna().reset_index()

def top_3_langs_ubuntu(ubuntu):
    '''
    This function takes in ubuntu and returns a Pandas Series containing the top 3 languages and their percentages.
    '''
    return ubuntu.language.value_counts(normalize=True).head(3)

def ubuntu_corpus(ubuntu):
    '''
    This function pulls in ubuntu and returns out a Pandas Series containing the words of the ubuntu linux corpus.
    '''
    ubuntu_corpus_list = []
    language = []
    for entry in range(len(ubuntu.readme_contents)):
        language.append(ubuntu.language[entry])
        ubuntu_corpus_list.extend(c.clean_data(ubuntu.readme_contents[entry]))
    return pd.Series(ubuntu_corpus_list)

def ubuntu_top_10_bigrams(ubuntu_corpus):
    '''
    This function takes in the ubuntu_corpus and returns out a Pandas Series containing the top 10 bigrams.
    '''
    return (pd.Series(nltk.ngrams(ubuntu_corpus, 2)).value_counts().head(10))

def ubuntu_top_50_bigram_wordcloud(ubuntu_corpus):
    '''
    This function takes in the ubuntu_corpus, finds the top 50 bigrams and their frequency and then displays a wordcloud of the bigrams.
    '''
    top_50_ubuntu_bigrams = (pd.Series(nltk.ngrams(ubuntu_corpus, 2)).value_counts().head(50))

    data = {k[0] + ' ' + k[1]: v for k, v in top_50_ubuntu_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.show()



def hypothesis_1_viz_1(master_df):
    '''
    This function takes in the master_df dataframe and displays a violinplot comparing the length of readmes for each distro.
    '''
    plt.figure(figsize=(12,7))
    sns.violinplot(data = master_df, y = 'length_of_readme', x = 'distro', palette = 'flare')
    plt.axhline(master_df.length_of_readme.mean(), color = '#0372CB')
    plt.ylim(0, 3500)
    plt.show()

def hypothesis_1_variables(master_df):
    '''
    This function sets variables to be used in the first hypothesis test and elsewhere.
    
    Returns:    alpha - the alpha value to be used in hypothesis testing
                ubuntu - a subset dataframe containing only info on the ubuntu repos.
                debian - a subset dataframe containing only info on the debian repos.
                arch - a subset dataframe containing only info on the arch repos.
                non_ubuntu - a subset dataframe containing only info on the non_ubuntu repos.
    '''
    # Setting alpha
    alpha = .05

    # Setting variables
    ubuntu = master_df[master_df.distro == 'ubuntu']
    debian = master_df[master_df.distro == 'debian']
    arch = master_df[master_df.distro == 'arch']
    non_ubuntu = master_df[master_df.distro != 'ubuntu']

    return alpha, ubuntu, debian, arch, non_ubuntu

def hypothesis_1_viz_2(non_ubuntu, ubuntu):
    '''
    This function takes in the non_ubuntu and ubuntu subset dataframes and does a histplot of the values to compare length of readmes
    across the two subsets.
    '''
    #Visualization of these results
    plt.figure(figsize=(18,8))
    sns.histplot(data = non_ubuntu.length_of_readme, bins = 290, color = 'red', label = 'non-ubuntu', stat = 'percent')
    sns.histplot(data = ubuntu.length_of_readme, bins = 274, label = 'ubuntu', stat = 'percent')
    plt.xlim(0, 3000)
    plt.legend()
    plt.show()

def hypothesis_2_variables(arch, debian, ubuntu):
    '''
    This function takes in the arch, debian and ubuntu subsets and finds how many repos use each of the following languages when split
    but specific distribution: Shell, Python, C, and other langauges.
    
    Returns:    observed - a dataframe containing number of instances of language by distribution
    '''
    # Setting corpus variables
    arch_language_nums = [len(arch.query("language == 'Shell'")), 
                        len(arch.query("language == 'Python'")), 
                        len(arch.query("language == 'C'")),
                        len(arch.query("language !='Shell' & language != 'Python' & language != 'C'"))]

    debian_language_nums = [len(debian.query("language == 'Shell'")), 
                        len(debian.query("language == 'Python'")), 
                        len(debian.query("language == 'C'")),
                        len(debian.query("language !='Shell' & language != 'Python' & language != 'C'"))]

    ubuntu_language_nums = [len(ubuntu.query("language == 'Shell'")), 
                        len(ubuntu.query("language == 'Python'")), 
                        len(ubuntu.query("language == 'C'")),
                        len(ubuntu.query("language !='Shell' & language != 'Python' & language != 'C'"))]
    observed = pd.DataFrame([arch_language_nums, debian_language_nums, ubuntu_language_nums], 
                            columns = ['Shell', 'Python', "C", "Other Langauges"], 
                            index = ['Arch', 'Debian', 'Ubuntu'])
    return observed

def hypothesis_2_viz(observed):
    '''
    This function takes in the observed dataframe and displays a stacked bar plot of the information in the dataframe.
    '''
    sns.set(style='white')
    observed.plot(kind= 'barh', stacked=True, color=['#1319FF', '#FF138F', '#FFA613', '#13FF83'], figsize = (18,8))
    plt.legend()
    plt.title('Langauges by Linux Distro', fontsize=16)
    plt.xlabel('Number of repos with Language')
    plt.ylabel('Linux Flavor')
    plt.show()
