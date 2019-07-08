import numpy as np
import pandas as pd

import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

training_path = "occupancy_data/datatraining.txt"
test_path = "occupancy_data/datatest.txt"
test2_path = "occupancy_data/datatest2.txt"


def pre_processing(path):
    data = pd.read_csv(path)
    data.drop('date', axis=1, inplace=True)
    data.drop('HumidityRatio', axis=1, inplace=True)

    values = data.values

    X, y = values[:, :-1], values[:, -1]

    return X, y


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
        model_ = GradientBoostingClassifier()
    elif model == 'RF':
        model_ = RandomForestClassifier(n_estimators=1250, n_jobs=4)
    elif model == 'Logistic':
        model_ = LogisticRegression(solver='lbfgs')
    else:
        model_ = LogisticRegression(solver='lbfgs')

    print("\nFit using \'{}\'\n".format(model))
    model_.fit(train_x, train_y)

    return model_


def evaluate_model(model_str):
    X, y = load_train()
    test_x, test_y = load_test()

    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=1)

    model = fit_model(train_x, train_y, model=model_str)

    y_hat_valid = model.predict(valid_x)
    y_hat_test = model.predict(test_x)

    # Calculate ROC-AUC score

    y_pred_prob = model.predict_proba(test_x)[:, 1]
    auc_score = roc_auc_score(test_y, y_pred_prob)

    # AUC with CV
    cv_scores = cross_val_score(model, X, y, cv=10)
    mean_cv = np.mean(cv_scores)

    validation_accuracy = accuracy_score(valid_y, y_hat_valid)
    test_accuracy = accuracy_score(test_y, y_hat_test)

    print("Validation Accuracy: {}\nTest Accuracy: {}\nROC-AUC Score: {}\nMean CV Scores: {}\n".format(
        validation_accuracy, test_accuracy, auc_score, mean_cv))


if __name__ == "__main__":
    help_str = "Selecta model to evaluation, choices are: "
    help_str += "\'Logistic\', \'GBT (Gradient Boost Tree)\', and \'RF (Random Forest)\'"

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=help_str)

    args = parser.parse_args()
    evaluate_model(args.model)
