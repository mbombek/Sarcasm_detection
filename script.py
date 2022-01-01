import json, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm


def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

#data = list(parse_data('./Sarcasm_Headlines_Dataset_v2.json'))

def decision_tree(X_cv_train, X_cv_test, y_train, y_test):
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree_model.fit(X_cv_train, y_train)

    #tree.plot_tree(tree_model)
    #plt.savefig('hey.pdf')
    #plt.show()

    y_pred = tree_model.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred))
    print('f1-score:', f1_score(y_test,y_pred))
    return f1_score(y_test,y_pred)

def logistic_regression(X_cv_train, X_cv_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_cv_train, y_train)
    logreg.score(X_cv_train, y_train)
    y_pred_logr = logreg.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred_logr)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred_logr))
    print('f1-score:', f1_score(y_test,y_pred_logr))
    return f1_score(y_test,y_pred_logr)

def support_vector_machine(X_cv_train, X_cv_test, y_train, y_test):
    model = svm.SVC(kernel='linear', gamma=1) 
    model.fit(X_cv_train, y_train)
    model.score(X_cv_train, y_train)
    y_pred_svm= model.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred_svm)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred_svm))
    print('f1-score:', f1_score(y_test,y_pred_svm))
    return f1_score(y_test,y_pred_svm)

def random_forest(X_cv_train, X_cv_test, y_train, y_test):
    forest = RandomForestClassifier (criterion='gini',
                                n_estimators=12, 
                                random_state=1)
    forest.fit(X_cv_train, y_train)
    forest.score(X_cv_train, y_train)
    y_pred_forest= forest.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred_forest)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred_forest))
    print('f1-score:', f1_score(y_test,y_pred_forest))
    return f1_score(y_test,y_pred_forest)

def naive_bayes(X_cv_train, X_cv_test, y_train, y_test):
    nb=MultinomialNB()
    nb.fit(X_cv_train, y_train)
    nb.score(X_cv_train, y_train)
    y_pred_nb= nb.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred_nb)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred_nb))
    print('f1-score:', f1_score(y_test,y_pred_nb))
    return f1_score(y_test,y_pred_nb)

def knn(X_cv_train, X_cv_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_cv_train, y_train)
    y_pred_knn= knn.predict(X_cv_test)
    test = np.array(y_test)
    predictions = np.array(y_pred_knn)
    print(confusion_matrix(test, predictions))
    print('accuracy:', accuracy_score(y_test,y_pred_knn))
    print('f1-score:', f1_score(y_test,y_pred_knn))
    return f1_score(y_test,y_pred_knn)



t1 = time.time()

df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json',lines=True)

#print(df.head())
#print(df.info())
#print(df.dtypes)
#print(df.is_sarcastic.value_counts())

X = df['headline']
y = df.is_sarcastic
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

cv = CountVectorizer(ngram_range=(1,3))
X_cv_train = cv.fit_transform(X_train)
X_cv_test = cv.transform(X_test)

print('Decision tree:\n')
dt_f1 = decision_tree(X_cv_train, X_cv_test, y_train, y_test)
print('LogReg:\n')
logreg_f1 = logistic_regression(X_cv_train, X_cv_test, y_train, y_test)
print('SVM:\n')
svm_f1 = support_vector_machine(X_cv_train, X_cv_test, y_train, y_test)
print('Random forest:\n')
forest_f1 = random_forest(X_cv_train, X_cv_test, y_train, y_test)
print('Naive Bayes:\n')
nb_f1 = naive_bayes(X_cv_train, X_cv_test, y_train, y_test)
print('kNN:\n')
knn_f1 = knn(X_cv_train, X_cv_test, y_train, y_test)

models = ['Decision Tree', 'Logistic regression', 'SVM', 'Random forest', 'Naive Bayes', 'kNN']
f1s = [dt_f1, logreg_f1, svm_f1, forest_f1, nb_f1, knn_f1]
data = {
    'Models': models,
    'F1 score': f1s
}
graph_df = pd.DataFrame(data)
graph_df = graph_df.sort_values(by=['F1 score'], axis=0, ascending=False)

t2 = time.time()
print('Time elapsed:' + str(t2 - t1))

fig, ax = plt.subplots()
sns.barplot(x=graph_df['Models'], y=graph_df['F1 score'], data=graph_df)
plt.savefig('model_comparison.png')
plt.show()

'''
fig = plt.figure()
plt.bar(models, f1s)
plt.xlabel('Models')
plt.ylabel('F1 score')
plt.savefig('model_comparison.png')
plt.show()
'''

