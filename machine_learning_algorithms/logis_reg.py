from sklearn.linear_model import LogisticRegression
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# for visualization
from sklearn.metrics import plot_confusion_matrix
p = [
     {'C' : range(1,10), 'solver' : ['sag'], 'penalty' : ['l2']}, #'max_iter' : range(1000, 3000)},
     {'C' : range(1,10), 'solver' : ['newton-cg', 'lbfgs'], 'penalty' : ['l2']},# 'max_iter' : range(1000, 3000)},
     {'C' : range(1,10), 'solver' : ['liblinear'], 'penalty' : ['l1', 'l2']}# 'max_iter' : range(1000, 3000)},
     #{'C' : range(1,10), 'solver' : ['saga'], 'penalty' : ['l1', 'l2'], 'max_iter' : range(1000, 3000)},
     #{'C' : range(1,10), 'solver' : ['saga'], 'penalty' : ['elasticnet'], 'l1_ratio' : range(0,1), 'max_iter' : range(1000, 3000)},
    ]

def log_normal(x_train, x_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_grid = ms.GridSearchCV(estimator = log_reg, param_grid = p, scoring = 'accuracy', cv = 4)
    log_grid_search = log_grid.fit(x_train, y_train)
    plot_confusion_matrix(log_grid, x_test, y_test)
    return print("Best parameters are: \n", log_grid_search.best_params_, "\n", "best accuracy score is: \n", log_grid_search.best_score_), plt.savefig("logis_reg_normal_confusion.png"), plt.show()

def log_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    pca_log = LogisticRegression()
    gs_pca = ms.GridSearchCV(estimator = pca_log, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_pca = gs_pca.fit(x_train2, y_train)
    plot_confusion_matrix(gs_pca, x_test2, y_test)
    return print("the score for pca is ", gs_search_pca.best_score_, "\n", "Best parameters are ", gs_search_pca.best_params_, "\n"), plt.savefig("logis_reg_pca_confusion.png"), plt.show()

def log_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis(n_components = n)
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    lda_log = LogisticRegression()
    gs_lda = ms.GridSearchCV(estimator = lda_log, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_lda = gs_lda.fit(x_train3, y_train)
    plot_confusion_matrix(gs_lda, x_test3, y_test)
    return print("the score for lda is ", gs_search_lda.best_score_, "\n", "Best parameters are ", gs_search_lda.best_params_, "\n"), plt.savefig("logis_reg_lda_confusion.png"), plt.show()