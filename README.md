# Maskininlarning-Yuna-Liu


Choosing the right estimator[ from sklearn][link]:

[link]: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


<img src="figures/ml_map.png" alt="Choosing the right estimator" />


https://github.com/kokchun/Maskininlarning-AI21/blob/main/Lectures/L3-overfitting-underfitting.ipynb
trai|validation|test:
- 1. fit models with train data, and test with validation data
- 2. run the 1st step for 100 times to get best hyperparameters degree d
- 3. train the model using choosen degree(from 2nd step) to **train+validation dataset**, and then predict on test data


https://github.com/kokchun/Maskininlarning-AI21/blob/main/Lectures/Lec10-RandomForest.ipynb
GridSearch():
- 1. train test split. 
- 2. Use GridSearch() to choose the best hyperparameters. Here only train data used, no validation data needed. 
- 3. Predict on test data. It is able to compare metrics for different types of models. Best model is then selected.