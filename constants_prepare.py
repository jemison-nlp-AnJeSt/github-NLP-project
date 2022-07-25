# Constants of prepare functions for the NLP project

#imports
import nltk
import unicodedata
import re
import json


def drop_nulls(df):
    """This function drops all nulls, as these are related to 
    repos missing a listed language."""
    df = df.dropna()
    return df


#____________________________________


ADDITIONAL_STOPWORDS = [
    'sudo',
    'distro',
    'linux',
    'aptget',
    'ubuntu',
    'debian',
    'arch',
    'archlinux',
    'git',
    'root',
    'image',
    'img',
    'instal',
    'use',
    'user',
    'packag',
    'file',
    'run',
    'system',
    'configur',
    'script',
    'set',
    'build',
    'need',
    'make',
    'option',
    'creat',
    'default'
]

def get_stopwords_from_file():
    """This function takes in the 5,000 top words that have a 1%
    difference or less of similarity between programming languages
    And adds these words onto the stopwords list"""
    with open('high_freq_stopwords.json', 'r') as f:
        add_stop_words_list = json.load(f) 
    return list(add_stop_words_list.values())+ ADDITIONAL_STOPWORDS


def clean_data(text):
    """This function cleans the text data (readme_content) for the 
    NLP project. It normalizes the words by adding additional stopwords,
    stemming, pulling out any odd symbols and https-related content"""
    ps = nltk.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english') + get_stopwords_from_file()
    text = (unicodedata.normalize('NFKD', str(text))
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[\(<\"]?http.*[\)>\"\s]', ' ', text).split()
    words = [re.sub(r'[^\w\s]', '', text) for text in words]
    words = [word for word in words if word!='']
    words = [word for word in words if word not in stopwords]
    words = [ps.stem(word) for word in words]
    words = [ps.stem(word) for word in words if word not in stopwords]
    return words   

def adding_columns(df):
    df['clean_readme'] = df.readme_contents.apply(clean_data)
    df['length_of_readme'] = df['readme_contents'].apply(lambda r : len(clean_data(r)))
    df['readme_string'] = df['clean_readme'].apply(lambda x: ','.join(map(str, x)))

    return df