import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("diabetes.csv")


dataset = pd.read_csv('diabetes.csv')


dataset.head()


dataset.shape


"""
#Dataset info

"""


dataset.info()


dataset.describe().T


"""
# **Count Null Values**
"""


dataset.isnull().sum()


"""
# Data Visualization

"""


sns.countplot(x= 'Outcome', data=dataset)


"""
# Histogram View
"""


import itertools

col = dataset.columns[:8]
plt.figure(figsize=(20, 15))
length = len(col)

for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length // 2) + (length % 2), 2, j + 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    dataset[i].hist(bins=20)
    plt.title(i)

plt.show()


"""
# Pair plot of dataset
"""


sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()


"""
# Heatmap of Dataset
"""


sns.heatmap(dataset.corr(), annot = True)
plt.show()


"""
# Data Processing
"""


dataset_new = dataset


dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)


dataset_new.isnull().sum()


"""
#Replacing NaN with mean values
"""


dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)


"""
# Statistical summary
"""


dataset_new.describe().T


"""
# Feature scalling using min max scaller
"""


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_new)


dataset_scaled = pd.DataFrame(dataset_scaled)


"""
# Selecting Features
"""


X = dataset_scaled.iloc[:,[1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values


"""
# Splitting X and Y
"""


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'])


"""
# Checking Dimensions
"""


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)


"""
# Data Modelling
"""


"""
# Logistic Regression
"""


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 42)
lr.fit(X_train, Y_train)
LogisticRegression(random_state=42)


"""
# Predicting rhe model
"""


training_accuracy = lr.score(X_train, Y_train)
print("Training Accuracy of Logistic Regression:", training_accuracy)


Y_pred_lr = lr.predict(X_test)


"""
# Classification Report
"""


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_lr))


from sklearn.metrics import f1_score
score_lr = round(accuracy_score(Y_test, Y_pred_lr)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
print('Precision: %.3f' % precision_score(Y_pred_lr, Y_test))
print('Reacll: %.3f' % recall_score(Y_test, Y_pred_lr))
print('F1 Score: %.3f' % f1_score(Y_test, Y_pred_lr))


"""
Confusion Matrix
"""


# prompt: Print confusion matrix from the given data after importing it from sklearn.matrics

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred_lr)
print(cm)



"""
# Heatmap of CM
"""


sns.heatmap(pd.DataFrame(cm), annot=True)


"""
# Plotting a graph for n_neighbors

"""


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

X_axis = list(range(1, 31))
acc = pd.Series()
x=range(1,31)

for i in list(range(1, 31)):
  knn_model = KNeighborsClassifier(n_neighbors = i)
  knn_model.fit(X_train, Y_train)
  prediction = knn_model.predict(X_test)
  acc = pd.concat([acc, pd.Series(metrics.accuracy_score(prediction, Y_test))])
plt.plot(X_axis, acc)
plt.xticks(x)
plt.title("Finding best value for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
print('Highest value: ',acc.values.max())


"""
# K nearest neighbors Algo
"""


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)


"""
# predicting KNN model
"""


knn.fit(X_train, Y_train)
Y_pred_knn=knn.predict(X_test)
score_knn = round(accuracy_score(Y_test, Y_pred_knn)*100, 2)


print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
print('Precision: %.3f' % precision_score(Y_test, Y_pred_knn))
print('Recall: %.3f' % recall_score(Y_test, Y_pred_knn))
print('F1 score: %.3f' % f1_score(Y_test, Y_pred_knn))


print(classification_report(Y_test, Y_pred_knn))


"""
# CM of knn
"""


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_knn)
print(cm)


sns.heatmap(pd.DataFrame(cm), annot=True)


"""
# SVC Algo
"""


svc_linear=SVC(kernel='linear')
svc_linear.fit(X_train, Y_train)
Y_pred_svc_linear = svc_linear.predict(X_test)
score_svc_linear = round(accuracy_score(Y_test, Y_pred_svc_linear)*100,2)
print("The accuracy score Achieved using Linear SVC is: "+str(score_svc_linear)+" %")


print('Precision: %.3f' % precision_score(Y_test, Y_pred_svc_linear))
print('Recall: %.3f' % recall_score(Y_test, Y_pred_svc_linear))
print('F1 score: %.3f' % f1_score(Y_test, Y_pred_svc_linear))


print(classification_report(Y_test, Y_pred_svc_linear))


"""
# Building SVC kernel = Polynomial
"""


svc_poly = SVC(kernel='poly')
svc_poly.fit(X_train, Y_train)
Y_pred_svc_poly = svc_poly.predict(X_test)


score_svc_poly = round(accuracy_score(Y_test, Y_pred_svc_poly)*100,2)
print("The accuracy score Achieved using Linear SVM is: "+str(score_svc_poly)+" %")


print('Precision: %.3f' % precision_score(Y_test, Y_pred_svc_poly))
print('Recall: %.3f' % recall_score(Y_test, Y_pred_svc_poly))
print('F1 score: %.3f' % f1_score(Y_test, Y_pred_svc_poly))


print(classification_report(Y_test, Y_pred_svc_poly))


"""
# Building the SVC kernel=Gaussian
"""


svc_gauss = SVC(kernel='rbf')
svc_gauss.fit(X_train, Y_train)
Y_pred_svc_gauss = svc_gauss.predict(X_test)


score_svc_gauss = round(accuracy_score(Y_test, Y_pred_svc_gauss)*100,2)
print("The accuracy score Achieved using Gaussian SVM is: "+str(score_svc_gauss)+" %")


print ('Precision: %.3f' % precision_score(Y_test, Y_pred_svc_gauss))
print ('Recall: %.3f' % recall_score(Y_test, Y_pred_svc_gauss))
print ('F1 score: %.3f' % f1_score(Y_test, Y_pred_svc_gauss))


print(classification_report(Y_test, Y_pred_svc_gauss))


"""
# SVC kernel = Sigmoid
"""


svc_sigmoid = SVC(kernel='sigmoid')
svc_sigmoid.fit(X_train, Y_train)
Y_pred_svc_sigmoid = svc_sigmoid.predict(X_test)
score_svc_sigmoid = round(accuracy_score(Y_test, Y_pred_svc_sigmoid)*100,2)
print("The accuracy score Achieved using Sigmoid SVM is: "+str(score_svc_sigmoid)+" %")


print(classification_report(Y_test, Y_pred_svc_sigmoid))


accuracy = {'svc_polynomial' : score_svc_poly, 'svc_linear' : score_svc_linear, 'svc_gaussian' : score_svc_gauss, 'svc_sigmoid' : score_svc_sigmoid}


model = list(accuracy.keys())
values = list(accuracy.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(model, values,width = 0.4)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy of different models")
plt.show()


"""
# Random Forest Algo
"""


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf.fit(X_train, Y_train)


Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_test, Y_pred_rf)*100,2)
print("The accuracy score Achieved using Random Forest is: "+str(score_rf)+" %")


print(classification_report(Y_test, Y_pred_rf))


"""
# Decision Tree Algo
"""


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, Y_train)


Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_test, Y_pred_dt)*100,2)
print("The accuracy score Achieved using Decision Tree is: "+str(score_dt)+" %")


"""
# ***Graph for Different Algo***
"""


accuracy_scores = [score_lr, score_knn, score_rf, score_dt]
algorithms = ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree']


plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracy_scores, color='skyblue')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy Score (%)')
plt.title('Accuracy Scores of Different Algorithms')
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
for i, score in enumerate(accuracy_scores):
    plt.text(i, score + 2, f'{score}%', ha='center', va='bottom')
plt.tight_layout()
plt.show()


"""
# Storing Model in pickle
"""


import pickle


pickle.dump(knn, open('models/diabetes_prediction_model_rf.pkl', 'wb'))


loaded_model = pickle.load(open('models/diabetes_prediction_model_rf.pkl', 'rb'))


"""
# Prediction By taking Input from user
"""


input_data1 = (137,138,43,33)
input_data1_as_numpy_array = np.asarray(input_data1)
input_reshape1 = input_data1_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_reshape1)
print(prediction)


if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


