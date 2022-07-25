import pandas as pd
import numpy as np


#visualizations:
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import GridspecLayout

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