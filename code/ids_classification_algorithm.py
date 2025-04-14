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
import matplotlib.pyplot as plt
from sklearn.preprocessing import maxabs_scale, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


if __name__ == "__main__":

    #Preprocessing
    df = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
    df_t = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)
    df = pd.concat([df_t, df],ignore_index=True)
    df.drop(df[df['attack_cat'] == 'Worms'].index, inplace=True)    #Worms drop
    
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
    dataframes = [X_test.iloc[:2000],X_test.iloc[2000:4000], X_test.iloc[4000:6000], X_test.iloc[6000:8000], X_test.iloc[8000:10000], X_test.iloc[10000:12000], X_test.iloc[12000:14000], X_test.iloc[14000:16000],X_test.iloc[16000:18000], X_test.iloc[18000:22000], X_test.iloc[22000:24000], X_test.iloc[24000:26000], X_test.iloc[26000:28000], X_test.iloc[28000:30000] ]
    tests = [Y_test.iloc[:2000],Y_test.iloc[2000:4000], Y_test.iloc[4000:6000], Y_test.iloc[6000:8000], Y_test.iloc[8000:10000], Y_test.iloc[10000:12000], Y_test.iloc[12000:14000], Y_test.iloc[14000:16000],Y_test.iloc[16000:18000], Y_test.iloc[18000:22000], Y_test.iloc[22000:24000], Y_test.iloc[24000:26000], Y_test.iloc[26000:28000], Y_test.iloc[28000:30000] ]    

    #CLASSIFICATION
    v = [df]*4
    v[0] = DecisionTreeClassifier(max_depth=12, min_samples_split=44)
    v[1] = RandomForestClassifier(max_depth=12, min_samples_split=44)
    v[2] = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(solver="adam", activation='relu', hidden_layer_sizes=(100,), max_iter=100))])
    v[3] = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier(n_neighbors=12, n_jobs=-1))])

    accuracy = 0
    n = 0
    model = VotingClassifier(estimators=[("DT", v[0]), ("RF", v[1]), ("MLP", v[2]), ("KNN", v[3])], voting= 'soft', n_jobs=4, weights=[3, 3, 1, 2])
    start_time = time.time()

    while accuracy < 85.3:
        model = model.fit(X_train, Y_train)
        testing_prediction = model.predict(X_train)
        accuracy = accuracy_score(Y_train, testing_prediction)*100
    print(f"Fitting time: {time.time() - start_time}")
    precision = 0
    prfs = 0

    
    for data in dataframes:

        start_time = time.time()

        testing_prediction = model.predict(data)
        print(f"Il df {n} è stato classificato in {time.time() - start_time}")
        accuracy = accuracy_score(tests[n], testing_prediction)*100

        result = pd.DataFrame(data)
        result.columns = df.columns[:-2]
        pred = pd.DataFrame(testing_prediction)
        pred.columns = ['attack_cat']
        result['proto'] = pd.to_numeric(result['proto'], downcast='integer')
        result['service'] = pd.to_numeric(result['service'], downcast='integer')
        result['state'] = pd.to_numeric(result['state'], downcast='integer')

        result['proto'] = le.inverse_transform(result['proto'])   
        result['service'] = le1.inverse_transform(result['service'])
        result['state'] = le.inverse_transform(result['state'])   
        pred['attack_cat'] = le3.inverse_transform(pred['attack_cat'])

        result = pd.concat([result['proto'],result['service'],result['dur'],result['rate']], axis=1)
        result = pd.concat([result.reset_index(drop=True), pred.reset_index(drop=True)], axis = 1)
        result.drop(result[result['attack_cat'] == 'Normal'].index, inplace=True)
        #result['id'] = pd.to_numeric(result['id'], downcast='integer')

        #REPORT PRODUCTION
        result.to_excel(f"FReport{n}.xlsx")
        print(f"\tIl df {n} è stato eseguito con una precisione del {accuracy}% in {time.time() - start_time}")
        
        precision = precision + accuracy
        n = n+1
    print(precision/n)
