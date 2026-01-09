import numpy as np
import matplotlib.pyplot as plt

def loadf_data(path):
    data=np.genfromtxt(path,delimiter=",", skip_header=1)
    X=data[:,:-1]
    y=data[:,-1]
    return X,y

def normalize(X):
    mean=X.mean(axis=0)
    std=X.std(axis=0)
    X_norm=(X-mean)/std
    return X_norm,mean,std

def train_test_split(X,y,test_size=0.2,seed=42):
    np.random.seed(seed)
    arr=np.random.permutation(len(X))
    test_count=int(len(X)*test_size)
    test_idx=arr[:test_count]
    train_idx=arr[test_count:]
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]

def predict(X,w,b):
    return np.dot(X,w)+b

def train_linear_regression(X,y,lr=0.01,epochs=2000):
    n_samples,n_features=X.shape
    w=np.random.rand(n_features)
    b=0
    for epoch in range(epochs):
        y_pred=predict(X,w,b)
        dw=(1/n_samples)* np.dot(X.T,(y_pred-y))
        db=(1/n_samples) * np.sum(y_pred-y)
        w-=lr*dw
        b-=lr*db
        if epoch%200==0:
            loss=np.mean((y_pred-y)**2)
            print(f"Epoch {epoch}, MSE: {loss:.4f}")


    return w,b

def mean_squared_error(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

if __name__=='__main__':
    X,y=loadf_data("House.csv")
    X_norm,mean,std=normalize(X)
    X_train,X_test,y_train,y_test=train_test_split(X_norm,y)
    w,b=train_linear_regression(X_train,y_train,lr=0.01,epochs=2000)
    print("Weights: ",w)
    print("Bias: ",b)
    y_pred=predict(X_test,w,b)
    mse=mean_squared_error(y_test,y_pred)
    print("Test MSE: ",mse)
    area_test_real=X_test[:,0]*std[0]+mean[0]
    plt.scatter(area_test_real,y_test,color='blue',label="Giá thật")
    plt.scatter(area_test_real,y_pred,color='red',label="Giá dự đoán")
    plt.xlabel("Area(m2)")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("result.png", dpi=300, bbox_inches="tight")
    plt.show()



