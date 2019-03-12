'''
    This script uses the small data set from UFC game website for attempt to simplify data.
    Models to try:
        logistic regression with L1 and L2 
        Random forest
        stacking
'''
import pandas as pd
import argparse
import glob
import numpy as np
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def fit_and_score_model(mdl, X_train, X_test, y_train, y_test):
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return pre, rec, acc


def stack_models(X_train, y_train, X, models):
    # import pdb; pdb.set_trace()
    stack_features = np.zeros((X.shape[0], len(models)))
    for i, mdl in enumerate(models):
        mdl.fit(X_train, y_train)
        stack_feature = mdl.predict(X)
        stack_features[:,i] = stack_feature
    X = np.append(X, stack_features, axis=1)
    return X


def prepare_small_data_set(game_data_dir, game_data_glob, match_data_path, EDA = False):
    match_df = pd.read_csv(match_data_path)
    small_dfs = []
    for di in glob.glob(game_data_dir + "/" + game_data_glob):
        small_dfs.append(pd.read_csv(di))
    game_df = pd.concat(small_dfs, axis = 0)
    if EDA:
        fig, ax = plt.subplots(2,2, figsize=(16, 12))
        # fig.subplots_adjust(bottom=0.4, top=1.5)
        ax = ax.ravel()
        for i, fea in enumerate(["Striking","Grappling","Stamina","Health"]):
            # import pdb; pdb.set_trace()
            game_df.hist(fea, ax = ax[i])
            ax[i].set_title(fea)
        plt.show()
    matches = match_df[["B_Name", "R_Name"]]
    match_df.winner.replace({"blue": 1, "red": 0}, inplace = True)
    blue_wins = match_df.winner
    X = []
    y = []
    for i in range(len(matches.B_Name)):
        if blue_wins[i] != 1 and blue_wins[i] != 0: continue
        blue_fighter_data = game_df[game_df['Player Name'] == matches.B_Name[i]].drop(columns = ['Player Name']).values
        if len(blue_fighter_data) == 0: continue
        red_fighter_data = game_df[game_df['Player Name'] == matches.R_Name[i]].drop(columns = ['Player Name']).values
        if len(red_fighter_data) == 0: continue
        try:
            # match = np.concatenate((blue_fighter_data[0], red_fighter_data[0]))
            match = blue_fighter_data[0] - red_fighter_data[0]
        except:
            import pdb; pdb.set_trace()
        X.append(match)
        y.append(blue_wins[i])
    # import pdb; pdb.set_trace()   
    return np.array(X), np.array(y)


def get_settings(X, y):
    logi_settings, rf_settings = [], []
    kf = KFold(n_splits = 5, shuffle = True, random_state = 7)
    kf.get_n_splits(X)
    logistic_params = {
        'C': [0.00001, 0.000025, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [1000, 3300, 10000, 33000, 100000]
    }
    rf_params = {
        'n_estimators': [25, 50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_features': ['auto', 'log2', None, 'sqrt'],
        'class_weight': ['balanced', None]
    }
    model_performance = {}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        RF = RandomForestClassifier(random_state = 1)
        optimized_rf = GridSearchCV(RF, rf_params, scoring = ['recall', 'precision', 'accuracy'], refit=False)
        optimized_rf.fit(X_train, y_train)
        print("RF highest recall: {} with params {}".format(np.max(optimized_rf.cv_results_['mean_test_recall']), optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_recall'])]))
        rf_settings.append(optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_recall'])])
        print("RF highest precision: {} with params {}".format(np.max(optimized_rf.cv_results_['mean_test_precision']), optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_precision'])]))
        rf_settings.append(optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_precision'])])
        print("RF highest accuracy: {} with params {}".format(np.max(optimized_rf.cv_results_['mean_test_accuracy']), optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_accuracy'])]))
        rf_settings.append(optimized_rf.cv_results_['params'][np.argmax(optimized_rf.cv_results_['mean_test_accuracy'])])
        
        Logi = LogisticRegression(random_state = 1)
        optimized_log = GridSearchCV(Logi, logistic_params, scoring = ['recall', 'precision', 'accuracy'], refit=False)
        optimized_log.fit(X_train, y_train)
        print("Logi highest recall: {} with params {}".format(np.max(optimized_log.cv_results_['mean_test_recall']), optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_recall'])]))
        logi_settings.append(optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_recall'])])
        print("Logi highest precision: {} with params {}".format(np.max(optimized_log.cv_results_['mean_test_precision']), optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_precision'])]))
        logi_settings.append(optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_precision'])])
        print("Logi highest accuracy: {} with params {}".format(np.max(optimized_log.cv_results_['mean_test_accuracy']), optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_accuracy'])]))
        logi_settings.append(optimized_log.cv_results_['params'][np.argmax(optimized_log.cv_results_['mean_test_accuracy'])])
    
    return logi_settings, rf_settings


def validate_settings(logi_settings, rf_settings, X, y):
    models = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    for i, logi_setting in enumerate(logi_settings):
        logi = LogisticRegression(random_state = 1, **logi_setting)
        precision, recall, accuracy = fit_and_score_model(logi, 
                                            X_train, 
                                            X_test, 
                                            y_train, 
                                            y_test)
        models["Logi" + str(i)] = {"precision": precision, "recall": recall, "params": logi_setting, "model": logi, "accuracy": accuracy}
        print("Logi" + str(i) + "  precision", precision, "recall", recall, "accuracy", accuracy, "params", logi_setting)

    for i, rf_setting in enumerate(rf_settings):
        rf = RandomForestClassifier(random_state = 1, **rf_setting)
        precision, recall, accuracy = fit_and_score_model(rf, 
                                            X_train, 
                                            X_test, 
                                            y_train, 
                                            y_test)
        models["RF" + str(i)] = {"precision": precision, "recall": recall, "params": rf_setting, "model": rf, "accuracy": accuracy}
        print("RF" + str(i) + "  precision", precision, "recall", recall, "accuracy", accuracy, "params", rf_setting)
    base_log = LogisticRegression(random_state = 1)
    precision, recall, accuracy = fit_and_score_model(base_log, 
                                    X_train, 
                                    X_test, 
                                    y_train, 
                                    y_test)
    models["base_log"] = {"precision": precision, "recall": recall, "params": "default", "model": base_log, "accuracy": accuracy}
    print("base_log   precision", precision, "recall", recall, "accuracy", accuracy, "params default")
    base_rf = RandomForestClassifier(random_state = 1)
    precision, recall, accuracy = fit_and_score_model(base_rf, 
                                    X_train, 
                                    X_test, 
                                    y_train, 
                                    y_test)
    models["base_rf"] = {"precision": precision, "recall": recall, "params": "default", "model": base_rf, "accuracy": accuracy}
    print("base_rf   precision", precision, "recall", recall, "accuracy", accuracy, "params default")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_data_dir")
    parser.add_argument("--game_data_glob")
    parser.add_argument("--match_data_path")
    args = parser.parse_args()
    X, y = prepare_small_data_set(args.game_data_dir, args.game_data_glob, args.match_data_path)
    # logi_settings, rf_settings = get_settings(X, y)
    # validate_settings(logi_settings, rf_settings, X, y)
    add_models = [
                        LogisticRegression(random_state = 1, C = 0.001, max_iter = 1000, solver = 'newton-cg'),
                        LogisticRegression(random_state = 1, C = 0.001, max_iter = 1000, solver = 'saga'),
                        LogisticRegression(random_state = 1, C = 1e-05, max_iter = 1000, solver = 'liblinear'),
                        RandomForestClassifier(random_state = 1, class_weight = None, criterion = 'entropy', max_features = 'auto', 
                                                min_samples_leaf = 2, n_estimators = 25)
                    ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    X_train_grid_search = stack_models(X_train, y_train, X_train, add_models)
    X_test_grid_search = stack_models(X_train, y_train, X_test, add_models)
    mdl_stack = LogisticRegression(random_state=1, penalty='l2', C=0.00001, class_weight='balanced')
    precision, recall, accuracy = fit_and_score_model(mdl_stack, 
                                        X_train_grid_search, 
                                        X_test_grid_search, 
                                        y_train, 
                                        y_test)
    print("precision", precision, "recall", recall, "accuracy", accuracy)


if __name__ == '__main__':
    main()