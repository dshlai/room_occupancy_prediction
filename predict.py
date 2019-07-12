import numpy as np
import pandas as pd

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.combine import SMOTEENN
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced

training_path = "occupancy_data/datatraining.txt"
test_path = "occupancy_data/datatest.txt"
test2_path = "occupancy_data/datatest2.txt"

RANDOM_STATE = os.environ.get('RANDOM_STATE', default=0)


def pre_processing(path):
    data = pd.read_csv(path)
    data.drop('date', axis=1, inplace=True)
    data.drop('HumidityRatio', axis=1, inplace=True)

    values = data.values

    X, y = values[:, :-1], values[:, -1]

    return X, y


def smoteenn_resample(x, y):

    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    X_resamp, y_resamp = smote_enn.fit_resample(x, y)

    return X_resamp, y_resamp


def load_train():
    train_x, train_y = pre_processing(training_path)
    test2_x, test2_y = pre_processing(test2_path)

    X = np.concatenate([train_x, test2_x], axis=0)
    y = np.concatenate([train_y, test2_y], axis=0)

    return X, y


def load_test():
    return pre_processing(test_path)


def fit_model(train_x, train_y, model=None):
    if model == 'GBT':
        model_ = GradientBoostingClassifier(random_state=RANDOM_STATE)

    elif model == 'BalancedGBT':
        model_ = pl.make_pipeline(SMOTEENN(random_state=RANDOM_STATE),
                                  GradientBoostingClassifier(random_state=RANDOM_STATE))

    elif model == 'RF':
        model_ = RandomForestClassifier(n_estimators=250,
                                        n_jobs=-1,
                                        random_state=RANDOM_STATE)
    elif model == 'BalancedRF':
        model_ = pl.make_pipeline(SMOTEENN(random_state=RANDOM_STATE),
                                  RandomForestClassifier(n_estimators=250,
                                                         n_jobs=-1,
                                                         random_state=RANDOM_STATE))

    elif model == 'Logistic':
        model_ = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)

    elif model == 'SVC':
        model_ = SVC(probability=True, gamma='scale', random_state=RANDOM_STATE)

    elif model == 'BalancedSVC':
        model_ = pl.make_pipeline(SMOTEENN(random_state=RANDOM_STATE),
                                  SVC(probability=True, gamma='scale', random_state=RANDOM_STATE))

    elif model == 'BalancedBag':
        model_ = BalancedBaggingClassifier(n_estimators=250, n_jobs=-1, random_state=RANDOM_STATE)

    else:
        raise RuntimeError("No Model specified")

    model_.fit(train_x, train_y)

    return model_


def make_simple_report(model_name, accu, balanced_accuracy_test, auc_score, mean_cv):

    model_names = {'Logistic': 'Logistic Regression',
                   'RF': 'Random Forest',
                   'BalancedRF': 'Balanced Random Forest',
                   'GBT': 'Gradient Boost Tree',
                   'BalancedGBT': 'Balanced Gradient Boost Tree',
                   'BalancedBag': 'Balanced Bagging',
                   'SVC': 'SVC',
                   'BalancedSVC': 'Balanced SVC'}

    model_name = model_names[model_name]

    report = pd.Series([model_name, accu, balanced_accuracy_test, auc_score, mean_cv],
                       index=['Model', 'Accuracy', 'Balanced Accuracy', 'AUC Score', 'Mean CV Score'])

    return report


def evaluate_model(model_str):
    X, y = load_train()
    test_x, test_y = load_test()

    X = np.concatenate([X, test_x], axis=0)
    y = np.concatenate([y, test_y], axis=0)

    s = StandardScaler()
    X = s.fit_transform(X)

    X, test_x, y, test_y = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = fit_model(X, y, model=model_str)

    y_hat = model.predict(test_x)

    # Calculate ROC-AUC score

    y_pred_prob = model.predict_proba(test_x)[:, 1]
    auc_score = roc_auc_score(test_y, y_pred_prob)

    # AUC with CV
    cv_scores = cross_val_score(model, X, y, cv=10)
    mean_cv = np.mean(cv_scores)

    # AccuracyScore
    accu = accuracy_score(test_y, y_hat)
    # Balanced Accuracy Score
    balanced_accuracy = balanced_accuracy_score(test_y, y_hat)

    simple = make_simple_report(model_str, accu, balanced_accuracy, auc_score, mean_cv)
    imb_report = classification_report_imbalanced(test_y, y_hat)

    return simple, imb_report


if __name__ == "__main__":
    help_str = "Select a model to evaluation, choices are: "
    help_str += "\'Logistic\', \'GBT (Gradient Boost Tree)\', and \'RF (Random Forest)\'"

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=help_str)

    args = parser.parse_args()
    evaluate_model(args.model)
