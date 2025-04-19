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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

if __name__ == "__main__":

    #Preprocessing
    df = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
    df_t = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)
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
    dataframes = [X_test.iloc[:2000],X_test.iloc[2000:4000], X_test.iloc[4000:6000], X_test.iloc[6000:8000], X_test.iloc[8000:10000], X_test.iloc[10000:12000], X_test.iloc[12000:14000], X_test.iloc[14000:16000],X_test.iloc[16000:18000], X_test.iloc[18000:22000], X_test.iloc[22000:24000], X_test.iloc[24000:26000], X_test.iloc[26000:28000], X_test.iloc[28000:30000] ]
    tests = [Y_test.iloc[:2000],Y_test.iloc[2000:4000], Y_test.iloc[4000:6000], Y_test.iloc[6000:8000], Y_test.iloc[8000:10000], Y_test.iloc[10000:12000], Y_test.iloc[12000:14000], Y_test.iloc[14000:16000],Y_test.iloc[16000:18000], Y_test.iloc[18000:22000], Y_test.iloc[22000:24000], Y_test.iloc[24000:26000], Y_test.iloc[26000:28000], Y_test.iloc[28000:30000] ]    

    #CLASSIFICATION
    v = [df]*6
    v[0] = DecisionTreeClassifier(max_depth=12, min_samples_split=44)
    v[1] = RandomForestClassifier(max_depth=12, min_samples_split=44)
    v[2] = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(solver="adam", activation='relu', hidden_layer_sizes=(100,), max_iter=100))])

    v[4] = MLPClassifier(solver="adam",alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1, learning_rate='adaptive')
    v[3] = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier(n_neighbors=12, n_jobs=-1))])

    accuracy = 0
    n = 0
    model = VotingClassifier(estimators=[("DT", v[0]), ("RF", v[1]), ("MLP", v[2]), ("KNN", v[3])], voting= 'soft', n_jobs=4, weights=[3, 3, 1, 2])
    model1 = VotingClassifier(estimators=[("DT", v[0]), ("RF", v[1]), ("MLP", v[4]), ("KNN", v[3])], voting= 'soft', n_jobs=4, weights=[3, 3, 1, 2])

    
    fitting_time = []
    start_time = time.time()
    v[0].fit(X_train, Y_train)
    fitting_time.append({time.time() - start_time})
    start_time = time.time()
    v[1].fit(X_train, Y_train)
    fitting_time.append({time.time() - start_time})
    start_time = time.time()
    v[2].fit(X_train, Y_train)
    fitting_time.append({time.time() - start_time})
    start_time = time.time()
    v[3].fit(X_train, Y_train)
    fitting_time.append({time.time() - start_time})

    start_time = time.time()
    model1 = model1.fit(X_train, Y_train)
    fitting_time.append({time.time() - start_time})

    while accuracy < 85.3:
        start_time = time.time()
        model = model.fit(X_train, Y_train)
        fitting_time.append({time.time() - start_time})
        testing_prediction = model.predict(X_train)
        accuracy = accuracy_score(Y_train, testing_prediction)*100
    print(f"Fitting time: {time.time() - start_time}")
    precision = 0
    prfs = 0

    #testing_time = []*6
    #s = 0
    #for data in dataframes:
#
    #    #start_time = time.time()
#
    #    for i  in range(0,4):
    #        s = time.time()
    #        v[i].predict(X_test)
    #        testing_time[i] = time.time() - s
    #    s = time.time()
    #    model1.predict(X_test)
    #    testing_time[4] = time.time() - s
    #    s = time.time()
    #    model.predict(X_test)
    #    testing_time[5] = time.time() - s
#
    #    #testing_prediction = model.predict(data)
    #    #print(f"Il df {n} è stato classificato in {time.time() - start_time}")
    #    accuracy = accuracy_score(tests[n], testing_prediction)*100
#
    #    result = pd.DataFrame(data)
    #    result.columns = df.columns[:-2]
    #    pred = pd.DataFrame(testing_prediction)
    #    pred.columns = ['attack_cat']
    #    result['proto'] = pd.to_numeric(result['proto'], downcast='integer')
    #    result['service'] = pd.to_numeric(result['service'], downcast='integer')
    #    result['state'] = pd.to_numeric(result['state'], downcast='integer')
#
    #    result['proto'] = le.inverse_transform(result['proto'])   
    #    result['service'] = le1.inverse_transform(result['service'])
    #    result['state'] = le.inverse_transform(result['state'])   
    #    pred['attack_cat'] = le3.inverse_transform(pred['attack_cat'])
#
    #    result = pd.concat([result['proto'],result['service'],result['dur'],result['rate']], axis=1)
    #    result = pd.concat([result.reset_index(drop=True), pred.reset_index(drop=True)], axis = 1)
    #    result.drop(result[result['attack_cat'] == 'Normal'].index, inplace=True)
    #    #result['id'] = pd.to_numeric(result['id'], downcast='integer')
#
    #    #REPORT PRODUCTION
    #    #result.to_excel(f"FReport{n}.xlsx")
    #    #print(f"\tIl df {n} è stato eseguito con una precisione del {accuracy}% in {time.time() - start_time}")
    #    
    #    precision = precision + accuracy
    #    n = n+1
    #print(precision/n)
    #print("Classification time: \n")
    #for i  in range(0,4):
    #    print(testing_time[i]/n)
    #s = time.time()
    #print(testing_time[4]/n)
    #print(testing_time[5]/n)
    #
    

    #FROM THIS POINT IS FOR THE WHOLE DATASET

    start_time = time.time()
    testing_prediction = []
    testing_time = []
    for i  in range(0,4):
        s = time.time()
        testing_prediction.append(v[i].predict(X_test))
        testing_time.append(time.time() - s)
    s = time.time()
    testing_prediction.append(model1.predict(X_test))
    testing_time.append(time.time() - s)
    s = time.time()
    testing_prediction.append(model.predict(X_test))
    testing_time.append(time.time() - s)
    #print(f"Il df {n} è stato classificato in {time.time() - start_time}")
    accuracy = []
    accuracy.append(accuracy_score(Y_test, testing_prediction[0])*100)
    accuracy.append(accuracy_score(Y_test, testing_prediction[1])*100)
    accuracy.append(accuracy_score(Y_test, testing_prediction[2])*100)
    accuracy.append(accuracy_score(Y_test, testing_prediction[3])*100)
    accuracy.append(accuracy_score(Y_test, testing_prediction[4])*100)
    accuracy.append(accuracy_score(Y_test, testing_prediction[5])*100)


    report = []
    report.append(classification_report(Y_test, testing_prediction[0], output_dict=True))
    report.append(classification_report(Y_test, testing_prediction[1], output_dict=True))
    report.append(classification_report(Y_test, testing_prediction[2], output_dict=True))
    report.append(classification_report(Y_test, testing_prediction[3], output_dict=True))
    report.append(classification_report(Y_test, testing_prediction[4], output_dict=True))
    report.append(classification_report(Y_test, testing_prediction[5], output_dict=True))

    precision = []
    recall = []
    f1 = []
    for i in range(0,6):
        precision.append(report[i]['weighted avg']['precision'])
        recall.append(report[i]['weighted avg']['recall'])
        f1.append(report[i]['weighted avg']['f1-score'])
    print(f"\tIl df {n} è stato eseguito con una precisione del {accuracy}% in {time.time() - start_time}")
    
    #precision = precision + accuracy
    #n = n+1
    #print(accuracy)
    #print("Classification report: ", classification_report(Y_test, testing_prediction))
    matrix = []
    matrix.append(confusion_matrix(Y_test, testing_prediction[0]))
    matrix.append(confusion_matrix(Y_test, testing_prediction[1]))
    matrix.append(confusion_matrix(Y_test, testing_prediction[2]))
    matrix.append(confusion_matrix(Y_test, testing_prediction[3]))
    matrix.append(confusion_matrix(Y_test, testing_prediction[4]))
    matrix.append(confusion_matrix(Y_test, testing_prediction[5]))

    FAR = []
    for i in range(0,6):
        FP = matrix[i].sum(axis=0) - np.diag(matrix[i])
        FN = matrix[i].sum(axis=1) - np.diag(matrix[i])
        TP = np.diag(matrix[i])
        FN = FN.astype(float)
        TN = matrix[i].sum() - (FP + FN + TP)
        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = FP/(FP+TN)
        print('FPR: '+str(FPR))
        FAR.append(str('%.8f' % FPR[6]))

    fig, axs = plt.subplots(6)
    fig.suptitle('Comparison of Classifiers')

    x = ['DT', 'RF', 'MLP', 'KNN', '4-EN']
    #for j in range(0,4):
    #    axs[0].plot(x, accuracy)
    #    axs[1].plot(x, precision)
    #    axs[2].plot(x, recall)
    #    axs[3].plot(x, f1)
    #    axs[4].plot(x, FAR)
    #    axs[5].plot(x, testing_time)
    #
    #axs[0].set(xlabel='Classifiers', ylabel='Accuracy')
    #axs[1].set(xlabel='Classifiers', ylabel='Precision')
    #axs[2].set(xlabel='Classifiers', ylabel='Recall')
    #axs[3].set(xlabel='Classifiers', ylabel='F1-score')
    #axs[4].set(xlabel='Classifiers', ylabel='FAR')
    #axs[5].set(xlabel='Classifiers', ylabel='Testing time \n(s)')
    #for ax in axs.flat:
    #    ax.label_outer()

    print("ACCURACY\n")
    for i in range(0,6):
        print(accuracy[i])
    print("PRECISION\n")
    for i in range(0,6):
        print(precision[i])
    print("RECALL\n")
    for i in range(0,6):
        print(recall[i])
    print("F1\n")
    for i in range(0,6):
        print(f1[i])
    print("FAR\n")
    for i in range(0,6):
        print(FAR[i])
    print("FITTING TIME\n")
    for i in range(0,6):
        print(fitting_time[i])
    print("TESTING TIME\n")
    for i in range(0,6):
        print(testing_time[i])

    #Confusions = []
    #Confusions.append(ConfusionMatrixDisplay(matrix[0], display_labels={0,1}))
    #Confusions.append(ConfusionMatrixDisplay(matrix[1], display_labels={0,1}))
    #Confusions.append(ConfusionMatrixDisplay(matrix[2], display_labels={0,1}))
    #Confusions.append(ConfusionMatrixDisplay(matrix[3], display_labels={0,1}))
    #Confusions.append(ConfusionMatrixDisplay(matrix[4], display_labels={0,1}))

    #for i in range(0,5):
    #    Confusions[i].plot()
    #    #Confusions[i].setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    #    plt.show()
    
    classifier_names = ['DT', 'RF', 'MLP', 
                    'KNN', 'EN1', 'EN2']
    v[4] = model
    v[5] = model1
    classifiers = v
#
   # # Create the plot
   # plt.figure(figsize=(10, 8))
#
   # for clf, name in zip(classifiers, classifier_names):
   #     # Predict probabilities (important: use [:, 1] for binary classification)
   #     if hasattr(clf, "predict_proba"):
   #         y_score = clf.predict_proba(X_test)[:, 1]
   #     else:
   #         # Some models like SVMs may need decision_function
   #         y_score = clf.decision_function(X_test)
#
   #     # Compute ROC curve and ROC area
   #     fpr, tpr, _ = roc_curve(Y_test, y_score)
   #     roc_auc = auc(fpr, tpr)
#
   #     # Plot ROC curve
   #     plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
#
   # # Plot settings
   # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal
   # plt.xlim([0.0, 1.0])
   # plt.ylim([0.0, 1.05])
   # plt.xlabel('False Positive Rate')
   # plt.ylabel('True Positive Rate')
   # plt.title('ROC Curve Comparison')
   # plt.legend(loc="lower right")
   # plt.grid()
#
    # Number of classes
    n_classes = len(np.unique(Y_test))

    # Binarize the output
    y_test_bin = label_binarize(Y_test, classes=range(n_classes))
    # Plot
    plt.figure(figsize=(10, 8))

    for clf, name in zip(classifiers, classifier_names):
        # Get predicted scores/probabilities
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
        else:
            y_score = clf.decision_function(X_test)

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot the macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'{name} (macro AUC = {roc_auc["macro"]:.2f})', lw=2)

    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve (Macro Average)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.show()
