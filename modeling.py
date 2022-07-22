import pandas as pd
import json
import sklearn.metrics as metrics
from itertools import product

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import nltk
import re
import unicodedata

BASELINE_ACCURACY = 0.52

RAND_SEED = 1729

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
    for i in range(1, 25):
        outputs.append(make_knn_model(train, validate, i))
        outputs.append(make_decision_tree_model(train, validate, i))
    rand_forest_params = return_product(5, 5, 5)
    print('starting rf and et')
    for prod in rand_forest_params:
        #print('making rf and et')
        outputs.append(make_random_forest_model(train, validate, leaf=prod[0], depth=prod[1], trees = prod[2]))
        outputs.append(make_extra_trees_model(train, validate, leaf=prod[0], depth=prod[1], trees = prod[2]))
    print('finished rf and et')
    return pd.DataFrame(outputs)

def test_models(train, validate):
    outputs = []
    for i in range(1, 25):
        print(f"testing k = {i}")
        outputs.append(make_knn_model(train, validate, i))
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
    train, validate = train_test_split(df, stratify=df['language'], train_size = 0.6, random_state = RAND_SEED)
    validate, test = train_test_split(df, stratify=df['language'], train_size = 0.75, random_state = RAND_SEED)
    return train, validate, test

def make_X_y_df(train, val):
    tfid = TfidfVectorizer(max_features=500, ngram_range = (1, 4))
    X_train = tfid.fit_transform(train['cleaned_readme'])
    X_val = tfid.transform(val['cleaned_readme'])
    y_train = train[['repo', 'language']]
    y_val = val[['repo', 'language']]
    return X_train, y_train, X_val, y_val

def return_product(l, d, t):
    """
    makes a itertools object iterable for the random forest and extra trees models
    """
    #make the range sets
    leaf_vals = range(1,l)
    depth_vals = range(2,d)
    #make tree values starting at 100 and going up in steps of 50
    tree_values = range(100, t*100, 50)
    #make the cartesian product
    product_output = product(leaf_vals, depth_vals, tree_values)
    return product_output

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

def make_knn_model(train, validate, neighbors, baseline_acc = BASELINE_ACCURACY):
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn = knn.fit(X_train, y_train['language'])
    y_train['predicted'] = knn.predict(X_train)
    y_val['predicted'] = knn.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'KNNeighbors',
        'attributes':f'n_neighbors = {neighbors}',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_decision_tree_model(train, validate, depth, baseline_acc = BASELINE_ACCURACY):
    """
    Makes a decision tree model and returns a dictionary containing calculated accuracy metrics
    """
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #make and fit the model
    dt = DecisionTreeClassifier(max_depth = depth, random_state=RAND_SEED)
    dt = dt.fit(X_train, y_train['language'])
    #make predictions
    y_train['predicted'] = dt.predict(X_train)
    y_val['predicted'] = dt.predict(X_val)
    # calculate metrics
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Decision Tree Classifier',
        'attributes': f"max_depth = {depth}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_random_forest_model(train, validate, leaf, depth, trees, baseline_acc = BASELINE_ACCURACY):
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #make and fit the model
    rf = RandomForestClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    rf = rf.fit(X_train, y_train['language'])
    #make predictions
    y_train['predicted'] = rf.predict(X_train)
    y_val['predicted'] = rf.predict(X_val)
    # calculate metrics
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Random Forest Classifier',
        'attributes': f"leafs = {leaf} : depth = {depth} : trees = {trees}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_extra_trees_model(train, validate, leaf, depth, trees, baseline_acc = BASELINE_ACCURACY):
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #make and fit the model
    et = ExtraTreesClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    et = et.fit(X_train, y_train['language'])
    #make predictions
    y_train['predicted'] = et.predict(X_train)
    y_val['predicted'] = et.predict(X_val)
    # calculate metrics
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Extra Trees Classifier',
        'attributes': f"leafs = {leaf} : depth = {depth} : trees = {trees}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output