from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, classification_report
# for visualization
from sklearn.metrics import plot_confusion_matrix

def gaus_normal(x_train, x_test, y_train, y_test):
    gaus_nb = GaussianNB()
    gaus_nb.fit(x_train, y_train)
    y_pred_gaus = gaus_nb.predict(x_test)
    acc = accuracy_score(y_pred_gaus, y_test)
    error = mean_absolute_error(y_pred_gaus, y_test)
    #Roc = roc_auc_score(y_pred_gaus, y_test)
    rep = classification_report(y_pred_gaus, y_test)
    plot_confusion_matrix(gaus_nb, x_test, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc, error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error,"\n", "Classification report:\n",rep,"\n"), plt.savefig("gaus_normal_confusion.png"), plt.show(), plt.savefig("gaus_normal_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')

def gaus_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    pca_gaus = GaussianNB()
    pca_gaus.fit(x_train2, y_train)
    y_pred_pca = pca_gaus.predict(x_test2)
    acc = accuracy_score(y_pred_pca, y_test)
    error = mean_absolute_error(y_pred_pca, y_test)
    #Roc = roc_auc_score(y_pred_pca, y_test)
    rep = classification_report(y_pred_pca, y_test)
    plot_confusion_matrix(pca_gaus, x_test2, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc, error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error,"\n", "Classification report:\n",rep,"\n"), plt.savefig("gaus_pca_confusion.png"), plt.show(), plt.savefig("gaus_pca_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')


def gaus_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis()
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    lda_gaus = GaussianNB()
    lda_gaus.fit(x_train3, y_train)
    y_pred_lda = lda_gaus.predict(x_test3)
    acc = accuracy_score(y_pred_lda, y_test)
    error = mean_absolute_error(y_pred_lda, y_test)
    #Roc = roc_auc_score(y_pred_lda, y_test)
    rep = classification_report(y_pred_lda, y_test)
    plot_confusion_matrix(lda_gaus, x_test3, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc, error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error,"\n", "Classification report:\n",rep,"\n"), plt.savefig("gaus_lda_confusion.png"), plt.show(), plt.savefig("gaus_lda_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')
