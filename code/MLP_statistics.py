from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import maxabs_scale, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import Sequential
from keras import layers
from keras._tf_keras.keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.metrics import accuracy_score
from keras._tf_keras.keras.optimizers import Adam
from sklearn.feature_selection import SelectFdr



if __name__ == "__main__":

    #Preprocessing
    df = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
    df_t = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)
    #A = []
    #for i in range(0,175341):
    #    A.append(i + 8224)
    #df['id'] = pd.array(A, dtype = int)
    df = pd.concat([df_t, df],ignore_index=True)
    df.drop(df[df['attack_cat'] == 'Worms'].index, inplace=True)    #Worms drop
    #df.drop(['id'],axis=1, inplace=True)                            #id drops for classification reasons
    
    start_time = time.time()
    
    le = LabelEncoder()
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    df['proto']         = le.fit_transform(df['proto'])
    df['service']       = le1.fit_transform(df['service'])
    df['state']         = le2.fit_transform(df['state'])
    df['attack_cat']    = le3.fit_transform(df['attack_cat'])

    df = df.drop(columns=['is_ftp_login', 'ct_ftp_cmd', 'dwin', 'trans_depth', 'is_sm_ips_ports', 'response_body_len', 'swin'])

    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[:-2]], df['attack_cat'], test_size=0.3)

    df['proto'] = le.inverse_transform(df['proto'])    
    df['service'] = le1.inverse_transform(df['service'])
    df['state'] = le2.inverse_transform(df['state'])
    print(f"Total pre-processing time: {time.time() - start_time}")

    import warnings

    import matplotlib.pyplot as plt
    
    from sklearn import datasets
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    strategies = [
    {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (128, 64, 32), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100, 50), 'activation': 'tanh', 'solver': 'adam'},
    {'hidden_layer_sizes': (100,), 'activation': 'logistic', 'solver': 'sgd'},
    {'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'solver': 'adam'}
    ]

# Store results for comparison
results = []

# Iterate through the different strategies and train models
colors = ['red', 'orange', 'blue', 'green', 'purple', 'black', 'pink']
linestyles = [
    ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1)))
]
i = 0

fig, axs = plt.subplots(7)
fig.suptitle('Comparison of MLPClassifier')


plt.figure(figsize=(5, 2.7), layout='constrained')
j = 0
for strategy in strategies:
    accuracyResult = []
    precisionResult = []
    recallResult = []
    F1Result = []
    FARResult = []
    fittingTimeResult = []
    testingTimeResult = []

    for i in range(1, 11):
        
        print(f"Training model with configuration: {strategy} and max_iter= {i*50}")
    
        model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(**strategy, max_iter=i*50))])
        #model = MLPClassifier(**strategy)

        stime = time.time()
        model.fit(X_train, Y_train)
        fittingTime = time.time() - stime

        stime = time.time()
        y_pred = model.predict(X_test)
        testing_time = time.time() - stime

        accuracy = accuracy_score(Y_test, y_pred)
        report = classification_report(Y_test, y_pred, output_dict=True)
        cnf_matrix = confusion_matrix(Y_test, y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        print('FP: '+str(FP))
        FN = FN.astype(float)
        print('Fn: '+str(FN))
        TP = TP.astype(float)
        print('FN: '+str(FN))
        TN = TN.astype(float)
        print('TN: '+str(TN))
        # false positive rate
        FPR = FP/(FP+TN)
        print('FPR: '+str(FPR))
        # False negative rate
        FNR = FN/(TP+FN)
        print('FNR: '+str(FNR))
        FAR = str(FNR[6])

        #results.append({
        #    'Configuration': str(strategy),
        #    'Accuracy': accuracy,
        #    'Precision': report['weighted avg']['precision'],
        #    'Recall': report['weighted avg']['recall'],
        #    'F1 Score': report['weighted avg']['f1-score'],
        #    'FAR' : FAR,
        #    'Fitting time' : fittingTime,
        #    'Testing time': testing_time,
        #    'Epoch' : i*50
        #})
        accuracyResult.append(accuracy)
        precisionResult.append(report['weighted avg']['precision'])
        recallResult.append( report['weighted avg']['recall'])
        F1Result.append(report['weighted avg']['f1-score'])
        FARResult.append(FAR)
        fittingTimeResult.append(fittingTime)
        testingTimeResult.append(testing_time)
    x = [50,100,150,200,250,300,350,400,450,500]
    #plt.plot(x, accuracyResult, label=str(strategy), color=colors[i], linestyle=linestyles[i])
    j = j+1
    
axs[0].plot(x, accuracyResult)
axs[1].plot(x, precisionResult)
axs[2].plot(x, recallResult)
axs[3].plot(x, F1Result)
axs[4].plot(x, FARResult)
axs[5].plot(x, fittingTimeResult)
axs[6].plot(x, testingTimeResult)


fig.legend()
y_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'FAR', 'Fitting time', 'Testing time']
i = 0
for ax in axs.flat:
    ax.set(ylabel=y_labels[i])
    i = i + 1
    

#results_df = pd.DataFrame(results)
#print("\nComparison of MLPClassifier Strategies:")
#print(results_df)



#results_df.set_index('Configuration', inplace=True)
#results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(10, 6))
#plt.title('MLPClassifier Strategies Comparison')
#plt.ylabel('Scores')
#plt.xticks(rotation=45, ha='right')
#plt.tight_layout()



plt.show()
