import pandas as pd
from sklearn.model_selection import StratifiedKFold
from mnist.loader import MNIST
import numpy as np

if __name__ == '__main__':
    mnist = MNIST('../input')
    x_train, y_train = mnist.load_training() 
    #x_test, y_test = mnist.load_testing() 

    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    #x_test = np.asarray(x_test).astype(np.float32)
    #y_test = np.asarray(y_test).astype(np.int32)

    print(f'Shape of X Train is {x_train.shape}')

    df = pd.DataFrame(data=x_train)
    df["label"] = y_train
    df = df.sample(frac=1).reset_index(drop=True)
    df['kfold'] = -1

    kf = StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y_train)):
        df.loc[v_, 'kfold'] = f
    
    print(df.head())

    df.to_csv("../input/mnist_train_folds.csv", index=False)

    print("File Created")
