# COMP0087
Git Repo for UCL's COMP0087 Statistical NLP coursework

## Proposal:

### Research Question/Hypothesis: 

A study on the decoder depth in an LSTM encoded task for Multi-Task Learning in NLP.

Aka: We have multiple tasks (1 main and then aux. tasks) and we encode them all with an LSTM. Then we “play” around with some CNNs to decode those tasks and see which one is better for what tasks.

### Previous Research/Novelty:
We will study where the heads of the tasks should be branched in the network for an optimal performance. Also, how shared layers affect the performance of these tasks. The auxiliary tasks would be Amazon reviews on different kinds of products, and the main task movie review.

### Papers:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9076160

### Possible Datasets:
For auxiliary tasks: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
For main task: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
