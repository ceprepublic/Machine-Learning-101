from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
p = [{'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'], 'max_features' : ['auto', 'sqrt', 'log2']}
    ]

def dec_normal(x_train, x_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt_grid = ms.GridSearchCV(estimator = dt, param_grid = p, scoring = 'accuracy', cv = 4)
    dt_grid_search = dt_grid.fit(x_train, y_train)
    return print("Best parameters are: \n", dt_grid_search.best_params_, "\n", "best accuracy score is: \n", dt_grid_search.best_score_)

def dec_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    pca_tr = DecisionTreeClassifier()
    gs_pca = ms.GridSearchCV(estimator = pca_tr, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_pca = gs_pca.fit(x_train2, y_train)
    return print("Best parameters are: \n", gs_search_pca.best_params_, "\n", "best accuracy score is: \n", gs_search_pca.best_score_)


def dec_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis(n_components = n)
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    lda_dt = DecisionTreeClassifier()
    gs_lda = ms.GridSearchCV(estimator = lda_dt, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_lda = gs_lda.fit(x_train3, y_train)
    return print("Best parameters are: \n", gs_search_lda.best_params_, "\n", "best accuracy score is: \n", gs_search_lda.best_score_)
