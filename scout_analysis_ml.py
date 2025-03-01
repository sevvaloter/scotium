# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import graphviz

from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


attributes = pd.read_csv("/Users/sevvalhaticeoter/PycharmProjects/scotium/scoutium_attributes.csv", delimiter= ";")
potential_labels = pd.read_csv("/Users/sevvalhaticeoter/PycharmProjects/scotium/scoutium_potential_labels.csv", delimiter=";")

attributes.head()
"""
#task_response_id:Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id :ilgili maçın idsi
# evaluator_id :değerlendirici(scoutun) idsi
# player_id : oyuncunun idsi
# position_id  : oyuncunun maçta oynadığı pozun idsi
# analysis_id :  Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value :Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
"""

potential_labels.head()
#potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)
# merge two dataframes om multiple columns
df = pd.merge(left= attributes, right= potential_labels, how= "left", on= ["task_response_id",  "match_id" , "evaluator_id" ,"player_id"])

df.head()
df.shape
df.dtypes
df.tail()
df.isnull().sum()
df.describe()


# checkin missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

df = df[df["position_id"] > 1]


df = df[df["potential_label"] != "below_average"]



new_df = pd.pivot_table(data=df,
                        index=['player_id', 'position_id', 'potential_label'],
                     columns=['attribute_id'],
                       values='attribute_value'
                            )
new_df = new_df.reset_index()
new_df.head()

new_df.columns = new_df.columns.map(str)
new_df.info()

label_encoder = LabelEncoder()
new_df.potential_label = label_encoder.fit_transform(new_df.potential_label)
new_df.potential_label.sample(10)


num_cols = new_df.columns[3:]
def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(new_df, col, plot=True)


scaler = StandardScaler()

new_df[num_cols] = scaler.fit_transform(new_df[num_cols])
new_df.head()

y = new_df["potential_label"]
X = new_df.drop(["potential_label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state= 42,use_label_encoder)
xgboost.set_config(**new_config)

models = [('LR', LogisticRegression()),
          ('RF', RandomForestClassifier()),
          ('SVR', SVC()),
          ('GBM', GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier()),
          ("LightGBM", LGBMClassifier()),
          ("CatBoost", CatBoostClassifier(verbose=False))]

def model_results(models):
    base_results = pd.DataFrame()
    base_results.index = ["accuracy_score", "f1_score", "roc_auc_score", "recall_score"]

    for name, classifier in models:
        cv_results = cross_validate(classifier, X_train, y_train, cv=10,
                                    scoring=["accuracy", "f1", "roc_auc", "recall"])
        base_results[name] = [cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(),
                              cv_results['test_roc_auc'].mean(), cv_results['test_recall'].mean()]

    base_results = base_results.T.sort_values("f1_score", ascending=False)

    return base_results


base_result = model_results(models)

base_result

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 200],
                  "colsample_bytree": [0.7, 1]}

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


classifiers = [('RF', RandomForestClassifier(), rf_params),
          ('GBM', GradientBoostingClassifier(), gbm_params),
          ("XGBoost", XGBClassifier(), xgboost_params),
          ("LightGBM", LGBMClassifier(), lgbm_params),
          ("CatBoost", CatBoostClassifier(verbose=False), catboost_params)]


def model_results(classifiers):
    model_grid = pd.DataFrame()
    model_grid.index = ["accuracy_score", "f1_score", "roc_auc_score", "recall_score"]

    best_models = {}
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X_train, y_train, cv=3,
                                    scoring=["accuracy", "f1", "roc_auc", "recall"])
        model_grid[name] = [cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(),
                            cv_results['test_roc_auc'].mean(), cv_results['test_recall'].mean()]

        best_models[name] = final_model
    model_grid = model_grid.T.sort_values("f1_score", ascending=False)

    return model_grid, best_models


final_model, best_model_params = model_results(classifiers)

final_model

base_result

best_model_params
""" {'RF': RandomForestClassifier(max_depth=5, max_features=5, min_samples_split=20,
                       n_estimators=500), 'GBM': GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, subsample=0.5), 'XGBoost': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
              colsample_bynode=None, colsample_bytree=0.7,
              enable_categorical=False, gamma=None, gpu_id=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_delta_step=None, max_depth=8,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=500, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=None, reg_alpha=None,
              reg_lambda=None, scale_pos_weight=None, subsample=None,
              tree_method=None, validate_parameters=None, verbosity=None), 'LightGBM': LGBMClassifier(colsample_bytree=1, learning_rate=0.01, n_estimators=1000), 'CatBoost': <catboost.core.CatBoostClassifier object at 0x7fc2a0de82e0>}

"""

final_rf = RandomForestClassifier(max_depth=5, max_features=5, min_samples_split=20,
                       n_estimators=500).fit(X_train, y_train)
y_pred = final_rf.predict(X_test)
y_prob = final_rf.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_prob)
print(classification_report(y_test, y_pred))

"""
              precision    recall  f1-score   support
           0       0.84      1.00      0.91        64
           1       1.00      0.33      0.50        18
    accuracy                           0.85        82
   macro avg       0.92      0.67      0.71        82
weighted avg       0.88      0.85      0.82        82

"""

# feature importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_lgbm, X)

