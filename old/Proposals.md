# Proposal:

## Research Question/Hypothesis: 

A study on the decoder depth in an LSTM encoded task for Multi-Task Learning in NLP.

Aka: We have multiple tasks (1 main and then aux. tasks) and we encode them all with an LSTM. Then we “play” around with some CNNs to decode those tasks and see which one is better for what tasks.

## Previous Research/Novelty:
We will study where the heads of the tasks should be branched in the network for an optimal performance. Also, how shared layers affect the performance of these tasks. The auxiliary tasks would be Amazon reviews on different kinds of products, and the main task movie review.

## Papers:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9076160

## Possible Datasets:
For auxiliary tasks: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
For main task: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


## Challenges/Risks:
Team fighting!!!!!!!! 

## Implementation:
Outline: 
1) Implement the LSTM and get it working for all datasets.
2) Create multiple heads for each tasks and add them to the model
3) Publish a paper and get 100%


# Previous Proposals:
## Proposal 1

### Research Question/Hypothesis: 

Is it possible to predict the outcome of Supreme Court Cases (using NLP)?

### Previous Research/Novelty:
Previous papers that we found that focus on legal prediciton use cases form the European Supreme Court

### Papers:
https://www.researchgate.net/publication/310615519_Predicting_Judicial_Decisions_of_the_European_Court_of_Human_Rights_A_Natural_Language_Processing_Perspective

### Possible Datasets:
Supreme Court Argument Transcripts: https://www.supremecourt.gov/oral_arguments/argument_transcript/2021 
Case Data and Justice Centered Data https://scdb.wustl.edu/data.php

### Challenges/Risks:
Parsing transcripts from the website might take a while to build a dataset

### Implementation:
Two possible approaches for this taks: 
1) Using oral arguments to predict who the winner is (petitioner/respondent) 
2) Predicting individual Justice behaviour given justice and case dataset type



## Proposal 2

### Research Question/Hypothesis: 
Perhaps investigate the effectiveness of transfer learning between datasets for use as an ML-powered autocomplete feature/recommendation system. Look at the effects of pretraining? Vary the architecture?

### Previous Research/Novelty:
There are loads of papers implementing BERT models( e.g., https://arxiv.org/pdf/1904.06690.pdf, https://aclanthology.org/2020.ecnlp-1.8.pdf, etc.), just need to find a novelty aspect to research. 

### Papers:
https://towardsdatascience.com/build-your-own-movie-recommender-system-using-bert4rec-92e4e34938c5 (https://github.com/CVxTz/recommender_transformer/blob/main/notebooks/inference.ipynb)

### Possible Datasets:
https://jmcauley.ucsd.edu/data/amazon/ - Amazon Product Datasets (e.g., music, movies, etc.)
https://cseweb.ucsd.edu/~jmcauley/datasets.html - Recommender Systems and Personalization Datasets (e.g., recipes)

### Challenges/Risks:
Original research question.

### Implementation:
Implement a transformer based recommender architecture (i.e., BERT model)
