import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


def fixOutliers(df, feature, to_do, low_boundary=500, high_boundary=0):
    """
    Remove outliers
    df: original dataframe
    feature: the variable choosed to filter
    function: choose 'low' to remove low outliers
            choose 'high' to remove high outliers
            choose 'both' to remove both outliers
    low_boundary: for the intention to compare with low whisker
    high_boundary: for the intention to compare with high whisker

    """
    "Remove the low outliers or weight under 54.6cm if the low whisker is lower than 54.6cm"
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    low_whisker = Q1 - 1.5 * IQR
    high_whisker = Q3 + 1.5 * IQR
 
    if to_do=='low':
        if low_whisker <= low_boundary:
            df = df[df[feature] >= low_whisker]
        else: 
            df = df[df[feature] >= low_boundary]
    elif to_do == 'high':
            if high_boundary > high_whisker:
                df = df[df[feature] < high_boundary]
            else: 
                df = df[df[feature] < (Q3 + 1.5 * IQR)]
    else:
        print('Hi, please run to_do = low first and then high in the second round.')   
    
    return df



def cardioProp(df, feature):
    """
    return a new dataframe to save the proportion of cardio disease among feature
    df: dataframe
    feature: the feature want to sort
    df_feature: cardio proportion among feature
    
    """
    list_feature=[feature,'cardio']
    df_feature = df[list_feature].groupby(feature).sum()
    df_feature['n_feature']=df[list_feature].groupby(feature).count()
    df_feature['cardio_%']=(df_feature['cardio']/df_feature['n_feature']*100).round(2)

    return df_feature




def box_plot(df, *feature):
    """boxplot of several features for df"""
    list_feature=feature

    fig, axes = plt.subplots(1, len(list_feature), dpi=120, figsize=(12,6))
    
    for ax, feature in zip(axes, list_feature):
        sns.boxplot(data=df[feature], ax=ax)
        ax.set(xlabel=feature)

    fig.tight_layout()
    fig.suptitle(f"Boxplot and outliers of {list_feature}", y=1.03, fontweight="bold");




def getWhisker(df, feature):
    """print the lower whisker and upper whisker of feature in df """
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    print(f"the lower whisker of {feature}: {Q1 - 1.5 * IQR:.2f}, the upper whisker of {feature}: {Q3 + 1.5 * IQR:.2f}")


def train_val_test_split(df, response_variable):
    """ use train_test_split twice to obtain train|val|test split """
    X, y = df.drop(response_variable, axis=1), df[response_variable] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
    return X_train, X_val, X_test, y_train, y_val, y_test


def getEvaMetrics(df_metrics, dataset, model, df_val_X, df_val_y):
    """
    1. print classification report
    2. plot confusion matrix
    3. calculate the recall and precision rate
    4. Return a new evaluation metrics by adding a new row: dataset, model, recall, precision
    
    df_metrics: the metrics matrix
    dataset: the choosen dataset
    model: the choosen model
    df_val_X: the validation df used as prediction
    df_val_y: the validation df used as comparision to prediction
    
    """
    y_pred = model.predict(df_val_X)
    
    print(classification_report(df_val_y, y_pred))

    cm = confusion_matrix(df_val_y, y_pred)
    ConfusionMatrixDisplay(cm).plot()

    #get tp, tp_and_fn and tp_and_fp w.r.t all classes
    tn, fp, fn, tp = cm.ravel()
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?msclkid=e1fb3d73abb311eca626f27972c45ef5

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    #add new row to end of metrics DataFrame
    df_metrics.loc[len(df_metrics.index)] = [dataset, model, recall, precision]
    return df_metrics
