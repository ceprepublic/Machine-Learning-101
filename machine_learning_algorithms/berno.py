from sklearn.naive_bayes import BernoulliNB
from sklearn import model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, classification_report
# for visualization
from sklearn.metrics import plot_confusion_matrix

def berno_normal(x_train, x_test, y_train, y_test):
    brn_nb = BernoulliNB()
    brn_nb.fit(x_train, y_train)
    y_pred_brn = brn_nb.predict(x_test)
    acc = accuracy_score(y_pred_brn, y_test)
    error = mean_absolute_error(y_pred_brn, y_test)
    rep = classification_report(y_pred_brn, y_test)
    plot_confusion_matrix(brn_nb, x_test, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc, error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error,"\n", "Classification report:\n",rep,"\n"), plt.savefig("berno_normal_confusion.png"), plt.show(), plt.savefig("berno_normal_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')


def berno_pca(x_train, x_test, y_train, y_test, n):
    pca = PCA(n_components = n)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.fit_transform(x_test)
    pca_brn = BernoulliNB()
    pca_brn.fit(x_train2, y_train)
    y_pred_pca = pca_brn.predict(x_test2)
    acc = accuracy_score(y_pred_pca, y_test)
    error = mean_absolute_error(y_pred_pca, y_test)
    #Roc = roc_auc_score(y_pred_pca, y_test)
    rep = classification_report(y_pred_pca, y_test)
    plot_confusion_matrix(pca_brn, x_test2, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc, error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error, "\n", "Classification report:\n", rep ,"\n"), plt.savefig("berno_pca_confusion.png"), plt.show(), plt.savefig("berno_pca_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')


def berno_lda(x_train, x_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis()
    x_train3 = lda.fit_transform(x_train, y_train)
    x_test3 = lda.transform(x_test)
    lda_brn = BernoulliNB()
    lda_brn.fit(x_train3, y_train)
    y_pred_lda = lda_brn.predict(x_test3)
    acc = accuracy_score(y_pred_lda, y_test)
    error = mean_absolute_error(y_pred_lda, y_test)
    #Roc = roc_auc_score(y_pred_lda, y_test)
    rep = classification_report(y_pred_lda, y_test)
    plot_confusion_matrix(lda_brn, x_test3, y_test)
    plt.figure(figsize = (12, 5))
    result = [acc,error]
    label = ["Accuracy", "Error"]
    colors=[ 'red', 'green']
    return print("Accuracy Score:\n", acc,"\n", "Mean Absolute Error:\n", error, "\n", "Classification report:\n", rep, "\n"), plt.savefig("berno_lda_confusion.png"), plt.show(), plt.savefig("berno_lda_dist.png"), plt.bar(label, result, color = colors, edgecolor='black')
