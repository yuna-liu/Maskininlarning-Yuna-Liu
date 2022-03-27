import joblib
import pandas as pd

# Reload joblib
myBestModel = joblib.load('cardio_disease_predictor')

# import data 
df_100_cat1 = pd.read_csv('data/test_samples.csv', index_col=False)
df_100_cat1 = df_100_cat1.drop(columns=['Unnamed: 0'])

# use only the X part of df_100_cat1
X_100 = df_100_cat1.drop("cardio", axis=1)

# predict the probability
prediction_prob= myBestModel.predict_proba(X_100)
prediction_prob = pd.DataFrame(prediction_prob, columns=['probability class 0', 'probability class 1'])
prediction_prob

# use the model go predict prediction
prediction = pd.DataFrame(myBestModel.predict(X_100), columns=['prediction'])
prediction

# merge the results
prediction_result = pd.concat([prediction_prob, prediction], axis=1)
prediction_result

#Save the predicted values
prediction_result.to_csv("data/prediction.csv", index=False)