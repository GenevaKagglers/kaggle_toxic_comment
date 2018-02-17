import os
from sklearn.externals import joblib
import pandas as pd
import defines
from utils import *

# # Naive Bayes
# sub_name = 'naiveBayes_ovr.csv'
# model = os.path.join(defines.MODEL_DIR, 'MultinomialNB.pkl')

# # LogisticRegression
# sub_name = 'logisticRegressionL2_ovr.csv'
# model = os.path.join(defines.MODEL_DIR, 'LogisticRegressionL2.pkl')

# # LogisticRegression TFIDFv2
# sub_name = 'logisticRegressionL2_tfidfv2.csv'
# model = os.path.join(defines.MODEL_DIR, 'LogisticRegressionL2_tfidfv2_auc.pkl')

# # LogisticRegression TFIDFv3
# sub_name = 'logisticRegressionL2_tfidfv3.csv'
# model = os.path.join(defines.MODEL_DIR, 'LogisticRegressionL2_tfidfv3_auc.pkl')

# # RandomForest TFIDFv2
# sub_name = 'randomForest_tfidfv2.csv'
# model = os.path.join(defines.MODEL_DIR, 'RandomForest_30_0p02_100_tfidfv2.pkl')

# RandomForest TFIDFv2
sub_name = 'logisticRegression_test7.csv'
model = os.path.join(defines.MODEL_DIR, 'Pipeline_test7.pkl')

# Load the test set
# vectorizer = fit_vectorizer()
vectorizer = joblib.load(os.path.join(defines.MODEL_DIR, 'Vectorizer_test7.pkl'))
ids, X_test = load_submission_set(vectorizer=vectorizer)

clf = joblib.load(model)
y_pred = np.array(clf.predict_proba(X_test))
df_out = pd.DataFrame(np.concatenate((ids.reshape(-1, 1), y_pred), axis=1), columns=['id'] + defines.CLASS_COLS)


# df_out = pd.DataFrame(columns=['id'] + defines.CLASS_COLS)
# df_out['id'] = ids

# for i, col in enumerate(defines.CLASS_COLS):
#
#     # Load the model
#     clf = pickle.load(open(model.format(col), 'rb'))
#
#     # Build predictions
#     # y_pred = np.array(clf.predict_proba(X_test))
#     y_pred = np.array(clf.predict(X_test))
#
#     df_out[col] = y_pred

df_out.to_csv(os.path.join(defines.SUB_DIR, sub_name), index=False)
