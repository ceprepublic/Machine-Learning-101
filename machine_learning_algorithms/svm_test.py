from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
# for visualization
from sklearn.metrics import plot_confusion_matrix
p = [{'C' : [1,2,3,4,5], 'kernel' : ['linear']},
     {'C' : [1,2,3,4,5], 'kernel' : ['rbf'], 'gamma' : [1, 0.5, 0.1, 0.01, 0.001]},
     {'C' : [1,2,3,4,5], 'kernel' : ['poly'], 'degree' : [1,2,3,4,5], 'gamma' : [1, 0.5, 0.1, 0.01, 0.001]}  
    ]

def svc_normal(x_train, x_test, y_train, y_test):
    svc = SVC()
    gs = ms.GridSearchCV(estimator = svc, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search = gs.fit(x_train, y_train)
    plot_confusion_matrix(gs, x_test, y_test)
    return print("Best parameters are ",gs_search.best_params_,"where the score is ", gs_search.best_score_), plt.savefig("svm_normal_confusion.png"), plt.show()


def svc_with_pca(x_train, x_test, y_train, y_test, n) :
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    svc_pca = SVC()
    gs_pca = ms.GridSearchCV(estimator = svc_pca, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_pca = gs_pca.fit(x_train2, y_train)
    plot_confusion_matrix(gs_pca, x_test2, y_test)
    return print("the score for pca is ", gs_search_pca.best_score_, "\n", "Best parameters are ", gs_search_pca.best_params_, "\n"), plt.savefig("svm_pca_confusion.png"), plt.show()


def svc_with_lda(x_train, x_test, y_train, y_test, n):
    # data consist of [x_train, x_test, y_train, y_test], a = indirgeme sayısı
    lda = LDA(n_components = n)
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    svc_lda = SVC()
    gs_lda = ms.GridSearchCV(estimator = svc_lda, param_grid = p, scoring = 'accuracy', cv = 4)
    gs_search_lda = gs_lda.fit(x_train3, y_train)
    plot_confusion_matrix(gs_lda, x_test3, y_test)
    return print("the score for lda is ", gs_search_lda.best_score_, "\n", "Best parameters are ", gs_search_lda.best_params_, "\n"), plt.savefig("svm_lda_confusion.png"),plt.show()
    
