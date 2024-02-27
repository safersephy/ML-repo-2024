from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()

X = bc.data
y = bc.target

def grid_search_binary_classifiers(X, y, metric='accuracy'):
    classifiers = {
        RandomForestClassifier(): {'n_estimators': [10, 100], 'max_features': ['sqrt', 'log2']},
        AdaBoostClassifier(): {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        GradientBoostingClassifier(): {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        LogisticRegression(): {'penalty': ['l2'], 'C': [1, 10]},
        SVC(probability=True): {'C': [1, 10], 'kernel': ['linear', 'rbf']},
        DecisionTreeClassifier(): {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        KNeighborsClassifier(): {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        GaussianNB(): {}
    }

    scoring_functions = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': 'roc_auc'
    }

    best_score = 0
    best_classifier_results = {}

    for i, (clf, params) in enumerate(classifiers.items()):
        print(f"Processing {i+1}/{len(classifiers)}: {clf.__class__.__name__}...")
        grid_search = GridSearchCV(clf, params, cv=5, scoring=scoring_functions[metric])
        grid_search.fit(X, y)



        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        current_score = grid_search.best_score_

        print(f"Evaluating {i+1}/{len(classifiers)}: {clf.__class__.__name__} - Best Parameters {grid_search.best_params_}, Best Score {grid_search.best_score_}")        


        if current_score > best_score:
            best_score = current_score
            best_classifier_results = {
                'Classifier': str(clf),
                'Best Parameters': grid_search.best_params_,
                'Best Score': current_score
            }

    print("\nBest Classifier Results based on", metric, ":")
    for key, value in best_classifier_results.items():
        print(f"{key}: {value}")

    return best_model, best_classifier_results

model, results = grid_search_binary_classifiers(X, y, metric='precision')
