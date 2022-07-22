import pandas as pd
import json
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import nltk
import re
import unicodedata

BASELINE_ACCURACY = 0.52

### FLOW CONTROL FUNCTIONS

def make_model_dfs(df):
    df = filter_languages(df)
    df['cleaned_readme'] = df.readme_contents.apply(clean_data)
    df = drop_nulls(df)
    train, validate, test = split_data(df)
    return train, validate, test

def model_maker(train, validate):
    outputs = []
    outputs.append(make_log_reg_model(train, validate))
    return pd.DataFrame(outputs)

### UTILITY FUNCTIONS

def get_stopwords_from_file():
    with open('high_freq_stopwords.json', 'r') as f:
        add_stop_words_list = json.load(f)
    return list(add_stop_words_list.values())

def clean_data(text):
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
    return ' '.join(words)

def filter_languages(df):
    keep_lang = df.language.value_counts().nlargest(10).index.tolist()
    df.loc[~(df.language.isin(keep_lang)), 'language'] = 'other'
    return df

def drop_nulls(df):
    return df.dropna()

def split_data(df):
    train, validate = train_test_split(df, stratify=df['language'], train_size = 0.6)
    validate, test = train_test_split(df, stratify=df['language'], train_size = 0.75)
    return train, validate, test

def make_X_y_df(train, val):
    tfid = TfidfVectorizer()
    X_train = tfid.fit_transform(train['cleaned_readme'])
    X_val = tfid.transform(val['cleaned_readme'])
    y_train = train[['repo', 'language']]
    y_val = val[['repo', 'language']]
    return X_train, y_train, X_val, y_val

### MODEL MAKERS

def make_log_reg_model(train, validate, baseline_acc = BASELINE_ACCURACY):
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    lm = LogisticRegression().fit(X_train, y_train['language'])
    y_train['predicted'] = lm.predict(X_train)
    y_val['predicted'] = lm.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'LogisticRegression',
        'attributes':'None',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_knn_model(train, validate, baseline_acc = BASELINE_ACCURACY):
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn = knn.fit(X_train, y_train['language'])
    y_train['predicted'] = knn.predict(X_train)
    y_val['predicted'] = knn.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'LogisticRegression',
        'attributes':'None',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output