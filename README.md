## Webscraping Linux Repos--Predicting the programming language.
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary 

In this NLP project, our team has scraped the top 3300 most-forked Linux-specific GitHub repositories to determine if we could build a model that can predict what programming language a repository  is using, based on the words contained with the README section. 

You can view our team's presentation deck here: <a href="https://www.canva.com/design/DAFHJ5GSN5g/9c1YRQnS7EsITQZRsDDr6w/view?utm_content=DAFHJ5GSN5g&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink">Linux: Grep Language Count</a>
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Initial Questions

> - How many unique words are there to each specific programming language?<br>
> - Are there any bigrams/trigrams that are specific to certain programming languages?<br>
> - Are there differences in words/phrases to Linux-flavors- specifically Ubuntu, Debian and Archlinux.<br>
> - Do certain programming languages have larger README sections than others? And if so, which ones?<br>
> - With Linux-flavors-Debian, Arch and Ubuntu-are there differences in README lengths? (ie does one flavor over the others seem to have more details needed or explained than others?)

#### Project Objectives
> - To build a function that can automatically scrape the top-forked Github repositories that are on Linux and the Linux-flavors: Ubuntu, Debian and Arch.<br>
> - To explore and find any differences of these repositories key words and the languages used within the repositories.<br>
> - To then build a model to that can predict what language a repository is using, based on the key words within the README sections. 
#### Data Dictionary
>
>
>
|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| repo | 2805 non-null: object | repository name/path |
| language | 2805 non-null: object | programming language used in repo |
| readme_contents| 2805 non-null: object | words and phrases within the readme |
| clean_readme| 2805 non-null: object | contents of readme after data has been cleaned/normalized|
| length_of_readme | 2805 non-null: int64 | how many words are in the repo |

##### Plan
> We plan to use the Codeup's webscraper acquire function and obtain our own GitHub tokens to webscrape the top-forked Linux repos.<br><br>
> We have decided to subdivide our search into three common Linux flavors: Ubuntu, Arch and Debian. We did this to also see if we could find any commonalities/difference between these flavors and the programming languages used.<br><br>
> Since we have 3300 scraped repositories, we will be normalizing these by:<br>
- 1) dropping nulls (as these are repositories that have no listed languages)
- 2) using NLTK tools to drop odd symbols, https-related content, tokenize the data to parse the words/grams more, and once our stopwords are found we will be stemming our entire corpus.
> Our exploration is first, centered around seeing any individual differences or commonalities of the three Linux flavors and then we combined these to for our overall main corpus to model on. <br><br>
> In exploration of the main corpus, we intend to find common words that we can include in our stopwords list (to get more finite in our search for predictably between programming languages).<br><br>
> We then will take what we have found in our main corpus and prepare the NLP dataset by using TF-IDF to turn the predictive words found into numbers/frequencies so that we can model using the new numbered data.<br><br>
> Once we find a viable Most Viable Product, we will work together to decide if anything else could be added to this model and then also determine if we would want to break the model down per Linux flavor that we acquired. 

#### Initial Hypotheses
> - **Hypothesis 1 -**
We believe that because these are Linux repositories, Shell will be the top programming language.

> - **Hypothesis 2 -** 
In differences of flavors, Archlinux will have more programming; less Shell than Ubuntu & Debian. 

> - **Hypothesis 3 -**
We have a theory that Ubuntu readme files may have more detail and length for new users.

> - **Hypothesis 4-**
Raspberry Pi will show up and be associated with Python language.


### Executive Summary - Conclusions & Next Steps
Using the data, we found there were significant differences between the repos. Specifically in languages, README size, and unique words.

In exploration, we found that Shell, Python and C were the most used languages. During this exploration phase, we also noticed large overlap in common words that we then took out almost 1,600 words by adding them to our stopwords list.

During the modeling stage, we found that Bagging Classifier using an underlying Decision Tree provided the best train/validate accuracy. Upon testing on out-of-sample-data, the model beat baseline by 13%. The model is overfitting which could be mitigated by changing how the data is vectorized and/or how the model is sampling the data for the estimators.

Our next steps would be to continue to experiment with hyperparameters, such as finding other common words to include in stopwords to find unique occurrences within the data acquired and to also pull more data for the model. We also would want to try changing the model to predict on the top 3 most used languages, versus our current model that predicts on the top 10.


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>


### Reproduce Our Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

In order to reproduce this project, you will need all the necessary files listed below to run the final project notebook. 
- [x] Read this README.md
- [ ] Have loaded all common DS libraries
- [ ] Download our acquired data stored in data folder [arch_linux_data.json, debian_data.json, high_freq_stopwords.json, master_df.json, ubuntu_data.json]
- [ ] Download all helper function files [constants_prepare.py, explore.py, moddel.py]
- [ ] Scrap notebooks (if desired, to dive deeper) are available in individual_preparation_notebooks folder
- [ ] Run the final report
