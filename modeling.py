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
from sklearn.ensemble import BaggingClassifier

import nltk
import re
import unicodedata

#baseline accuracy
BASELINE_ACCURACY = 0.52

#taxicab number
RAND_SEED = 1729

#these are used if there is no stop words file
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
    """
    Filters the languages, makes a clean readme column to pass, drops nulls and splits the data
    """
    #filter to top x languages
    df = filter_languages(df)
    #make the clean readme column
    df['cleaned_readme'] = df.readme_contents.apply(clean_data)
    #drop nulls
    df = drop_nulls(df)
    #split the data into subsets
    train, validate, test = split_data(df)
    return train, validate, test

def model_maker(train, validate, baseline_acc = BASELINE_ACCURACY):
    """
    Makes a mass of models and returns a dataframe with accuracy metric
    """
    outputs = []
    #make logistic regression model
    outputs.append(make_log_reg_model(train, validate, baseline_acc))
    #make knn and decision trees
    for i in range(1, 25):
        outputs.append(make_knn_model(train, validate, i, baseline_acc))
        outputs.append(make_decision_tree_model(train, validate, i, baseline_acc))
    rand_forest_params = return_product(3, 5, 4)
    print('starting rf and et')
    #make random forest and extra tree models
    for prod in rand_forest_params:
        #print('making rf and et')
        outputs.append(make_random_forest_model(train, validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
        outputs.append(make_extra_trees_model(train, validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
    print('finished rf and et')
    estimators = [
        {'model':LogisticRegression(), 'name':'LogisticRegression'},
        {'model':KNeighborsClassifier(), 'name':'KNeighborsClassifier'},
        {'model':DecisionTreeClassifier(), 'name':'DecisionTreeClassifier'},
        {'model':ExtraTreesClassifier(), 'name':'ExtraTreesClassifier'}
    ]
    #make ensemble model
    for estimator in estimators:
        outputs.append(make_bagging_classifier(train, validate, estimator['model'], estimator['name'], baseline_acc = baseline_acc))
    return pd.DataFrame(outputs)

def baseline_model_maker(train, validate):
    """
    Creates a baseline model and returns metrics on train and validate sets
    """
    #get the relevant columns
    baseline_model = train.loc[:, ['repo', 'language']]
    baseline_model_val = validate.loc[:, ['repo', 'language']]
    #prediction is the mode of the langauges
    baseline_model['predicted'] = train['language'].mode().to_list()[0]
    baseline_model_val['predicted'] = train['language'].mode().to_list()[0]
    #get the metrics
    metrics_dict = metrics.classification_report(train['language'], baseline_model['predicted'], output_dict=True, zero_division = True)
    metrics_dict_val = metrics.classification_report(validate['language'], baseline_model_val['predicted'], output_dict=True, zero_division = True)
    output = {
        'model':'Baseline Model',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
    }
    #return the metrics as a dataframe and the baseline accuracy for the model
    return pd.DataFrame([output]), metrics_dict['accuracy']

def test_model(train, validate, test, baseline_acc = BASELINE_ACCURACY):
    """
    Final model trained, and then run on unseen test data.  Dataframe of metrics returned
    """
    #split data into X an y
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    _, _, X_test, y_test = make_X_y_df(train, test)
    #make and train the model
    bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators = 20, max_samples = 0.5, max_features = 0.5, bootstrap=False, random_state = RAND_SEED).fit(X_train, y_train['language'])
    #get predictions
    y_train['predicted'] = bc.predict(X_train)
    y_val['predicted'] = bc.predict(X_val)
    y_test['predicted'] = bc.predict(X_test)
    #get the metrics
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    metrics_dict_test = metrics.classification_report(y_test['language'], y_test['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'BaggingClassifier',
        'attributes':f'estimator = DecisionTreeClassifier',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'test_accuracy':metrics_dict_test['accuracy'],
        'better_than_baseline':metrics_dict_test['accuracy'] > baseline_acc
    }
    #return the dataframe
    return pd.DataFrame([output])

### UTILITY FUNCTIONS

def get_stopwords_from_file():
    """
    Gets list of high frequency words
    """
    with open('data/high_freq_stopwords.json', 'r') as f:
        add_stop_words_list = json.load(f)
    return list(add_stop_words_list.values())

def clean_data(text):
    """
    Cleans the data for modeling
    """
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
    """
    filters to the top ten languages
    """
    #get list of languages to keep
    keep_lang = df.language.value_counts().nlargest(10).index.tolist()
    #all other languages are other
    df.loc[~(df.language.isin(keep_lang)), 'language'] = 'other'
    return df

def drop_nulls(df):
    """
    Drop nulls
    """
    return df.dropna()

def split_data(df):
    """
    Split data into subsets and stratify on target variable
    """
    train, validate = train_test_split(df, stratify=df['language'], train_size = 0.6, random_state = RAND_SEED)
    validate, test = train_test_split(df, stratify=df['language'], train_size = 0.75, random_state = RAND_SEED)
    return train, validate, test

def make_X_y_df(train, val):
    """
    vectorizes and makes X and y sets for the data
    """
    #vectorize
    tfid = TfidfVectorizer(max_features=500, ngram_range = (1, 4))
    #make x and y sets
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

def make_bagging_classifier(train, validate, estimator, estimator_name, baseline_acc = BASELINE_ACCURACY):
    """
    Makes a bagging classifier based on passed estimator
    """
    #make x and y sets
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #fit model
    bc = BaggingClassifier(base_estimator=estimator, n_estimators = 20, max_samples = 0.5, max_features = 0.5, bootstrap=False, random_state = RAND_SEED).fit(X_train, y_train['language'])
    #predict
    y_train['predicted'] = bc.predict(X_train)
    y_val['predicted'] = bc.predict(X_val)
    #get metrics
    metrics_dict = metrics.classification_report(y_train['language'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_val['language'], y_val['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'BaggingClassifier',
        'attributes':f'estimator = {estimator_name}',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_log_reg_model(train, validate, baseline_acc = BASELINE_ACCURACY):
    """
    makes a logistic regression model
    """
    #split into x and y sets
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #fit the model
    lm = LogisticRegression().fit(X_train, y_train['language'])
    #make predictions
    y_train['predicted'] = lm.predict(X_train)
    y_val['predicted'] = lm.predict(X_val)
    # get metrics
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
    """
    Make a knn model
    """
    #split the data
    X_train, y_train, X_val, y_val = make_X_y_df(train, validate)
    #fit the model
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn = knn.fit(X_train, y_train['language'])
    #make predictions
    y_train['predicted'] = knn.predict(X_train)
    y_val['predicted'] = knn.predict(X_val)
    #get metrics
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
    #split the data
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
    """
    Makes a random forest model
    """
    #split the data
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
    """
    makes an extra trees model
    """
    #split the data
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