# Constants for the NLP project

import nltk
import re
import unicodedata

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

def clean_data(text):
    ps = nltk.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
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