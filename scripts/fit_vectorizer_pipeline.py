import pickle
from utils import *
import defines
from textutils import *

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV


# Data
df_train = pd.read_csv(os.path.join(defines.DATA_DIR, 'train.csv'))
df_train['comment_text'].fillna('unknown', inplace=True)
X_train = df_train[defines.INPUT_COL].as_matrix()
y_train = df_train[defines.CLASS_COLS].as_matrix()

params = {'tfidf__max_df': [0.2, 0.4, 0.7],
          'tfidf__min_df': [2, 5],
          'tfidf__ngram_range': [(1, 2), (2, 2)],
          'tfidf__lowercase': [True, False],
          'tfidf__tokenizer': [LemmaTokenizer(), StemTokenizer(), None]
          }


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.4, min_df=2, stop_words='english', analyzer='word',
                             lowercase=False, ngram_range=(1, 2), tokenizer=StemTokenizer())

# clf = OneVsRestClassifier(
#         RandomForestClassifier(n_estimators=200,
#                                max_features=0.02,
#                                min_samples_leaf=100,
#                                class_weight='balanced_subsample',
#                                random_state=42,
#                                verbose=10,
#                                n_jobs=2)
#     )

clf = OneVsRestClassifier(
    LogisticRegressionCV(Cs=10, max_iter=400, class_weight='balanced', penalty='l2', dual=True, solver='liblinear',
                         fit_intercept=True, scoring='roc_auc', random_state=42, n_jobs=1, verbose=1)
    )

pipe = Pipeline(steps=[('tfidf', vectorizer),
                       ('transf', TfidfTransformer()),
                       ('clf', clf)])

grid = GridSearchCV(pipe, param_grid=params, scoring='roc_auc', error_score=0.5, n_jobs=2, verbose=10, cv=3)

grid.fit(X_train, y_train)

print('Best score: {0:.3f}'.format(grid.best_score_))

clf = grid.best_estimator_.named_steps['clf']

vectorizer = grid.best_estimator_.named_steps['tfidf']

print(vectorizer)
print(clf)


joblib.dump(vectorizer, os.path.join(defines.MODEL_DIR, 'Vectorizer_test.pkl'))
joblib.dump(clf, os.path.join(defines.MODEL_DIR, 'Pipeline_test.pkl'))
