import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


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
    list_feature=[feature,'cardio']
    df_feature = df[list_feature].groupby(feature).sum()
    df_feature['n_feature']=df[list_feature].groupby(feature).count()
    df_feature['cardio_%']=(df_feature['cardio']/df_feature['n_feature']*100).round(2)

    return df_feature




def box_plot(df, *feature):

    list_feature=feature

    fig, axes = plt.subplots(1, len(list_feature), dpi=120, figsize=(12,6))
    
    for ax, feature in zip(axes, list_feature):
        sns.boxplot(data=df[feature], ax=ax)
        ax.set(xlabel=feature)

    fig.tight_layout()
    fig.suptitle(f"Boxplot and outliers of {list_feature}", y=1.03, fontweight="bold");




def getWhisker(df, feature):
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    print(f"the lower whisker of {feature}: {Q1 - 1.5 * IQR:.2f}, the upper whisker of {feature}: {Q3 + 1.5 * IQR:.2f}")


def catBMI(x):
    if x < 18.5:
        return 'underweight'
    elif x>=18.5 & x<=25:
        return 'normal'
    elif x>25 & x<=30:
        return 'overweight'
    elif x>30 & x<=35:
        return 'obese(class I)'
    elif x>35 & x<=40:
        return 'obese(class II)'
    else:
        return 'obese(class III)'



def catBlood(vec):
    x = vec[0]
    y = vec[1]
    if x < 120 & y < 80:
        return 'healthy'
    elif x>=120 & x<130 & y < 80:
        return 'elevated'
    elif x>=130 & x<140 | y>=80 & y<90:
        return 'hypertension stage 1'
    elif x>=140 & x<180 | y>=90 & y<120:
        return 'hypertension stage 2'
    else:
        return 'hypertension crisis'
