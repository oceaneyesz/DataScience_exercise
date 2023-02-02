# <1155181315>
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Problem 2
def problem_2(filename,predictors,target):
    # write your logic here, model is the NN model
    batch_size, learning_rate = 10, 0.01
    csv_data = pd.read_csv(filename)
    X = csv_data[predictors].values
    y = csv_data[target].values
    torch.manual_seed(5726)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5726)
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
#     y=scaler.fit_transform(y)
    model = nn.Sequential(nn.Linear(in_features=11, out_features=5,bias=True), nn.ReLU(),nn.Linear(in_features=5, out_features=3,bias=True), nn.ReLU(),nn.Linear(in_features=3, out_features=1,bias=True), nn.Sigmoid())
    data_x = torch.FloatTensor(X_train)
    data_y = torch.Tensor(y_train).float()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses=[]
    results=[]
    for epoch in range(500):
        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    targets = model(torch.FloatTensor(X_test))
#     print(metrics.mean_squared_error(np.round(targets.detach().numpy()), y_test))
#     print(metrics.accuracy_score(np.round(targets.detach().numpy()), y_test))
    pred_y_test=targets.detach().numpy()

    test_precision=precision_score(y_test,np.round(pred_y_test),average=None)[1]
    test_recall=recall_score(y_test,np.round(pred_y_test),average=None)[1]
    return model, test_precision, test_recall

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# Problem 3
def problem_3(filename,predictors,target):
    # write your logic here, model is the RF model
    model, mean_cv_acc, sd_cv_acc = 0, 0, 0
    df = pd.read_csv(filename)
    df_getdummy=pd.get_dummies(data=df)
    X = df_getdummy[predictors]
    y = df_getdummy[target]
    scaler = StandardScaler()
    X_=scaler.fit_transform(X)
    y_=scaler.fit_transform([y])
    clf = RandomForestClassifier(random_state=5726)
    clf.fit(X_,np.ravel(y_))
    kfold = KFold(n_splits=8,shuffle=True, random_state=5726)
#     for train_index, test_index in kf.split(X):
# #     X_train, X_test = X[train_index], X[test_index]
# #     y_train, y_test = y[train_index], y[test_index]
#         X_ = X[train_index]
#         y_ = y[train_index]
    result = cross_val_score(clf, X_, y,scoring='accuracy', cv=kfold)
    mean_cv_acc=result.mean()
    sd_cv_acc=result.std()
    model = clf

    return model, mean_cv_acc, sd_cv_acc


from sklearn.svm import SVR
# Problem 4
def problem_4(filename,predictors,target):
    # write your logic here, model is the SVR model
    csv_data = pd.read_csv(filename)
    csv_data.head()
    TargetVariable = [target]
    Predictors = predictors
    X = csv_data[Predictors].values
    y = csv_data[TargetVariable].values
    torch.manual_seed(5726)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5726)
#     y_train = scaler.fit_transform(y_train)
#     y_test = scaler.fit_transform(y_test)
    model = SVR(kernel='poly')
    model.fit(X_train,np.ravel(y_train))
    y_pre = model.predict(torch.FloatTensor(X_test))
#     y_pre = y_pre.detach().numpy()
    test_mae = metrics.mean_absolute_error(y_test, y_pre)
    test_rmse = metrics.mean_squared_error(y_test, y_pre)**0.5
    #model, test_mae, test_rmse = 0, 0, 0

    return model, test_mae, test_rmse


from sklearn import linear_model
# Problem 5
def problem_5(filename,predictors,target):
    # write your logic here, model is the MLR model
    df = pd.read_csv(filename)
    df_getdummy=pd.get_dummies(data=df)
    X = df_getdummy[predictors]
    y = df_getdummy[target]
    torch.manual_seed(5726)
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X)
    y = np.array(y)
    y_ = scaler.fit_transform(y.reshape(-1, 1))
    regression = linear_model.LinearRegression()
    regression.fit(X_, y_)
    kfold = KFold(n_splits=8,shuffle=True, random_state=5726)   
    result=cross_val_score(regression,X_,y_, scoring='neg_mean_squared_error',cv=kfold)
    mean_cv_mse=abs(result.mean())
    sd_cv_mse=result.std()
    model = regression

    model, mean_cv_mse, sd_cv_mse = 0, 0, 0

    return model, mean_cv_mse, sd_cv_mse


from kneed import KneeLocator
from sklearn.cluster import KMeans
# Problem 6
def problem_6(train_filename,predictors,test_filename):
    # write your logic here, model is the k-mean model
    model, k, result = 0, 0, []

    
    csv_train=pd.read_csv(train_filename)
    csv_test=pd.read_csv(test_filename)
    train=csv_train[predictors].values
    test=csv_test[predictors].values
    OMP_NUM_THREADS=4
    scaler=StandardScaler()
    train = np.array(train)
    test = np.array(test)
    train=scaler.fit_transform(train)
    test=scaler.fit_transform(test)
    sse = [] 
    for k in range(1, 11):
        kmeans = KMeans(init='random',n_clusters=k, n_init=5, random_state=5726)
        kmeans.fit(train)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    k = kl.elbow
    model=KMeans(init='random', n_clusters=k, n_init=5, random_state=5726)
    model.fit(train)
    result=model.predict(test)


    return model, k, result