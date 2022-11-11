from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# for visualization
from sklearn.metrics import plot_confusion_matrix

p=[{'n_estimators':range(1,50),'criterion':['entropy']},
   {'n_estimators':range(1,50),'criterion':['gini']}]

def random_normal(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf_grid = ms.GridSearchCV(estimator=rf, param_grid=p, scoring='accuracy',cv=4)
    rf_grid_search = rf_grid.fit(x_train, y_train)
    plot_confusion_matrix(rf_grid, x_test, y_test)
    return print("Best parameters are: \n", rf_grid_search.best_params_, "\n", "best accuracy score is: \n", rf_grid_search.best_score_), plt.savefig("random_forest_normal_confusion.png") ,plt.show()

def random_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    rf_pca = RandomForestClassifier()
    gs_pca = ms.GridSearchCV(estimator = rf_pca, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_pca = gs_pca.fit(x_train2, y_train)
    plot_confusion_matrix(gs_pca, x_test2, y_test)
    return print("Best parameters are: \n", gs_search_pca.best_params_, "\n", "best accuracy score is: \n", gs_search_pca.best_score_), plt.savefig("random_forest_pca_confusion.png"), plt.show()

def random_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis(n_components = n)
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    rf_lda = RandomForestClassifier()
    gs_lda = ms.GridSearchCV(estimator = rf_lda, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_lda = gs_lda.fit(x_train3, y_train)
    plot_confusion_matrix(gs_lda, x_test3, y_test)
    return print("Best parameters are: \n", gs_search_lda.best_params_, "\n", "best accuracy score is: \n", gs_search_lda.best_score_), plt.savefig("random_forest_lda_confusion.png"), plt.show()

