from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# for visualization
from sklearn.metrics import plot_confusion_matrix
p = [{'n_neighbors' : range(1,10), 'weights' : ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric' : ['minkowski'], 'p' : range(2,5)},
     {'n_neighbors' : range(1,10), 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree'], 'leaf_size' : range(30,50)},
     {'n_neighbors' : range(1,10), 'weights' : ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric' : ['manhattan', 'euclidean']}
    ]

def knn_normal(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn_grid = ms.GridSearchCV(estimator = knn, param_grid = p, scoring = 'accuracy', cv = 4)
    knn_grid_search = knn_grid.fit(x_train, y_train)
    plot_confusion_matrix(knn_grid, x_test, y_test)
    return print("Best parameters are: \n", knn_grid_search.best_params_, "\n", "best accuracy score is: \n", knn_grid_search.best_score_), plt.savefig("neighbor_normal_confusion.png"), plt.show() 

def knn_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    pca_knn = KNeighborsClassifier()
    gs_pca = ms.GridSearchCV(estimator = pca_knn, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_pca = gs_pca.fit(x_train2, y_train)
    plot_confusion_matrix(gs_pca, x_test2, y_test)
    return print("the score for pca is ", gs_search_pca.best_score_, "\n", "Best parameters are ", gs_search_pca.best_params_, "\n"), plt.savefig("neighbor_pca_confusion.png"), plt.show()

def knn_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis(n_components = n)
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    lda_knn = KNeighborsClassifier()
    gs_lda = ms.GridSearchCV(estimator = lda_knn, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_lda = gs_lda.fit(x_train3, y_train)
    plot_confusion_matrix(gs_lda, x_test3, y_test)
    return print("the score for lda is ", gs_search_lda.best_score_, "\n", "Best parameters are ", gs_search_lda.best_params_, "\n"), plt.savefig("neighbor_lda_confusion.png"), plt.show()
