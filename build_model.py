from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, f1_score,precision_score,recall_score,accuracy_score, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import  make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import copy
import pandas as pd

model = None
knn = KNeighborsClassifier(n_neighbors= 8 , metric = 'l1')
lr = LogisticRegression(penalty='l2',C=1.0, max_iter=10000)
dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)
svc = SVC(kernel='poly', degree=3, max_iter=300000)
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators = 50, random_state=1, max_features=20)
adaBoost = AdaBoostClassifier(n_estimators=10, learning_rate=1, random_state=1)


def preprocess(X, y, cat) :
    new_data = copy.deepcopy(X)
    columns = X.columns
    columns_data = {}
    for column in columns:
        if new_data[column].dtype != int:
            columns_data[column] = new_data[column].unique()
            for i in range(len(columns_data[column])):
                new_data.loc[:, column][new_data[column] == columns_data[column][i]] = int(i)
            new_data[column] = pd.to_numeric(new_data[column])
    # print(new_data)
    y = LabelEncoder().fit_transform(y)
    X= new_data.iloc[:, 0:19]
    return X, y


def build(m, X, y, num, cat) :
    X, y = preprocess(X, y, cat)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if m=='knn' :
        model = knn
    elif m=='lr' :
        model = lr
    elif m=='dt' :
        model = dt
    elif m=='svc':
        model = svc
    elif m=='nb' :
        model = nb
    elif m=='adaBoost' :
        model = adaBoost
    else :
        model = rf

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return model, accuracy, precision, recall, f1
