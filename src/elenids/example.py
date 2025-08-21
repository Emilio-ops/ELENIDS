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
from importlib.resources import files
import shap
import pickle
import joblib


def load_testing():
    """
    This function returns the Dataframe of the UNSW_NB_15_testing-set
    """
    path = files('elenids').joinpath('data','UNSW_NB15_testing-set.csv')
    return pd.read_csv(path)

def load_training():
    """
    This function returns the Dataframe of the UNSW_NB_15_training-set
    """
    path = files('elenids').joinpath('data','UNSW_NB15_training-set.csv')
    return pd.read_csv(path)

def load():
    """
    This function returns the Dataframe of the complete dataset
    """
    df = load_testing()
    df_t = load_training()
    
    df = pd.concat([df_t, df],ignore_index=True)
    df.drop(df[df['attack_cat'] == 'Worms'].index, inplace=True)    #Worms drop
    df = df.drop(columns=['is_ftp_login', 'ct_ftp_cmd', 'dwin', 'trans_depth', 'is_sm_ips_ports', 'response_body_len', 'swin'])
    return df

def build():
    """
    This function returns the ensemble model composed of DT, RF, KNN, and MLP.

    Returns
    -------
    model : VotingClassifier
    """
    v = [0]*4
    v[0] = Pipeline([('scaler', MinMaxScaler()), ('DT', DecisionTreeClassifier(max_depth=12, min_samples_split=44))])
    v[1] = Pipeline([('scaler', MinMaxScaler()), ('RF', RandomForestClassifier(max_depth=12, min_samples_split=44))])
    v[2] = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(solver="adam", activation='relu', hidden_layer_sizes=(100,), max_iter=100))])
    v[3] = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier(n_neighbors=12, n_jobs=-1))])
    model = VotingClassifier(estimators=[("DT", v[0]), ("RF", v[1]), ("MLP", v[2]), ("KNN", v[3])], voting= 'soft', n_jobs=4, weights=[3, 3, 1, 2])
    return model

def Fastbuild():
    """
    This function returns the ensemble model composed of DT, RF, and MLP.
    The weight of the classifiers is derived from their F1-scores: 0.8305 0.8149 0.8414 .

    Returns
    -------
    model : VotingClassifier
    """
    v = [0]*3
    v[0] = Pipeline([('scaler', MinMaxScaler()), ('DT', DecisionTreeClassifier(max_depth=12, min_samples_split=44))])
    v[1] = Pipeline([('scaler', MinMaxScaler()), ('RF', RandomForestClassifier(max_depth=12, min_samples_split=44))])
    #v[0] = DecisionTreeClassifier(max_depth=12, min_samples_split=44)
    #v[1] = RandomForestClassifier(max_depth=12, min_samples_split=44)
    v[2] = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(solver="adam", activation='relu', hidden_layer_sizes=(100,), max_iter=100))])
    model = VotingClassifier(estimators=[("DT", v[0]), ("RF", v[1]), ("MLP", v[2])], voting= 'soft', n_jobs=4, weights=[3.05, 1.49, 4.14])
    return model


def savePickle(model, filename):
    """
    Save the model to a file using pickle.

    Parameters
    ----------
    model : object
        The model to be saved.

    filename : str
        The name of the file where the model will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def saveJoblib(model, filename):
    """
    Save the model to a file using joblib.

    Parameters
    ----------
    model : object
        The model to be saved.

    filename : str
        The name of the file where the model will be saved.
    """
    joblib.dump(model, filename, compress=3)

def loadPickle(filename):
    """
    Load a model from a file using pickle.

    Parameters
    ----------
    filename : str
        The name of the file from which the model will be loaded.

    Returns
    -------
    model : object
        The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def loadJoblib(filename):
    """
    Load a model from a file using joblib.

    Parameters
    ----------
    filename : str
        The name of the file from which the model will be loaded.

    Returns
    -------
    model : object
        The loaded model.
    """
    return joblib.load(filename)

def save(model, filename):
    """
    Save the model to a file.

    Parameters
    ----------
    model : object
        The model to be saved.

    filename : str
        The name of the file where the model will be saved.
    """
    '''
    pickle (and joblib and cloudpickle by extension), has many documented security vulnerabilities by design and 
    should only be used if the artifact, i.e. the pickle-file, is coming from a trusted and verified source. 
    You should never load a pickle file from an untrusted source, similarly to how you should never execute 
    code from an untrusted source.
    We are aware of this, but we haven't found a better way yet.
    '''
    try:
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        if filename.endswith('.pkl'):       #pickle format already specified
            savePickle(model, filename)
        elif filename.endswith('.joblib'):  #joblib format already specified
            saveJoblib(model, filename)
        else:
            try:
                savePickle(model, filename + '.pkl')
            except Exception as e:
                try:
                    saveJoblib(model, filename + '.joblib')
                except Exception as e:
                    raise ValueError(f"Failed to save model: {e}")
    except ValueError as e:
        print(f"Error: {e}")
        return

def load(filename):
    """
    Load a model from a file.

    Parameters
    ----------
    filename : str
        The name of the file from which the model will be loaded.

    Returns
    -------
    model : object
        The loaded model.
    """
    try:
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        if filename.endswith('.pkl'):       #pickle format already specified
            return loadPickle(filename)
        elif filename.endswith('.joblib'):  #joblib format already specified
            return loadJoblib(filename)
        else:
            try:
                return loadPickle(filename + '.pkl')
            except Exception as e:
                try:
                    return loadJoblib(filename + '.joblib')
                except Exception as e:
                    raise ValueError(f"Failed to load model: {e}")
    except ValueError as e:
        print(f"Error: {e}")
        return None

def fit(model, X_train, Y_train):
    """Fit the model according to X_train, Y_train.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Y_train : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
    accuracy = 0
    while accuracy < 85.3:
        model = model.fit(X_train, Y_train)
        testing_prediction = model.predict(X_train)
        accuracy = accuracy_score(Y_train, testing_prediction)*100
    return model


def classify(model, data,col, le,le1,le2,le3):
    """
    Perform classification on an array of test vectors data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input samples.

    col : array-like of shape (n_samples, n_features)
        The names of the columns of the final report.

    le : fitted label encoder (protocol).

    le1 : fitted label encoder (service).
    
    le2 : fitted label encoder (state).

    le3 : fitted label encoder (attack_category).


    Returns
    -------
    none

    Produces
    -------
    Excel file (.xlsx) containing all the classified attacks (if the data array contained at least an attack).
    """
    testing_prediction = model.predict(data)

    result = pd.DataFrame(data)
    result.columns = col
    pred = pd.DataFrame(testing_prediction)
    pred.columns = ['attack_cat']
    result['proto'] = pd.to_numeric(result['proto'], downcast='integer')
    result['service'] = pd.to_numeric(result['service'], downcast='integer')
    result['state'] = pd.to_numeric(result['state'], downcast='integer')

    result['proto'] = le.inverse_transform(result['proto'])   
    result['service'] = le1.inverse_transform(result['service'])
    result['state'] = le2.inverse_transform(result['state'])   
    pred['attack_cat'] = le3.inverse_transform(pred['attack_cat'])

    result = pd.concat([result['proto'],result['service'],result['dur'],result['rate']], axis=1)
    result = pd.concat([result.reset_index(drop=True), pred.reset_index(drop=True)], axis = 1)
    result.drop(result[result['attack_cat'] == 'Normal'].index, inplace=True)
    #result['id'] = pd.to_numeric(result['id'], downcast='integer')

    #REPORT PRODUCTION
    result.to_excel(f"FReport{time.time()}.xlsx")


def info(model, 
         data, 
         columns, 
         classifier_name = None):
    """
    Provide additional information about the decision making behind the classification of the data.

    Parameters
    ----------
     model : ELENIDS or Fast-ELENIDS.

    data : array-like of shape (n_samples, n_features)
        The input samples.

    columns : list of the names of the features
        In the original dataset.
    
    classifier_name : Name of the classifier that you'd like to
        Know more about. If None, informations about all the classifiers 
        Will be provided.

    Returns
    -------
    SHAP evaluations of the model.
    """
    shap.initjs()


    if classifier_name == None:
        for idx, estimator in enumerate(model.estimators_):
            print(f"\n=== SHAP for estimator {idx}: {type(estimator).__name__} ===")
            try:
                explainer = shap.Explainer(estimator.predict, data)
                shap_values = explainer.shap_values(data)
            except Exception as e:
                try:
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(data)
                except Exception as e:
                    try:
                        explainer = shap.KernelExplainer(estimator.predict, data)
                        shap_values = explainer.shap_values(data)
                    except Exception as e:
                        print(f"SKIPPING estimator {idx} ({type(estimator).__name__}): {e}")
                        continue

            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    print(f"Class {i} summary plot:")
                    shap.summary_plot(class_shap_values, data, max_display=10, show=False)
                    plt.title(f"Estimator {idx} - Class {i} SHAP Summary")
                    plt.show()
                    shap.summary_plot(class_shap_values, data, plot_type="bar", feature_names=columns, show=False)
                    plt.title(f"Estimator {idx} - Class {i} SHAP Bar")
                    plt.show()
            else:
                shap.summary_plot(shap_values, data, max_display=10, show=False)
                plt.title(f"Estimator {idx} - SHAP Summary")
                plt.show()
                shap.summary_plot(shap_values, data, plot_type="bar", feature_names=columns, show=False)
                plt.title(f"Estimator {idx} - SHAP Bar")
                plt.show()
    else:
        if classifier_name == 'DT':
            explainer = shap.TreeExplainer(model.estimators_[0])
            shap_values = explainer.shap_values(data)
            print(shap_values)
            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    print(f"Class {i} summary plot:")
                    shap.summary_plot(class_shap_values, data, max_display=10, show=False)
                    plt.title(f"Estimator DT - Class {i} SHAP Summary")
                    plt.show()
                    shap.summary_plot(class_shap_values, data, plot_type="bar", feature_names=columns, show=False)
                    plt.title(f"Estimator DT - Class {i} SHAP Bar")
                    plt.show()
            else:
                shap.summary_plot(shap_values, data, max_display=10, show=False)
                plt.title(f"Estimator DT - SHAP Summary")
                plt.show()
                shap.summary_plot(shap_values, data, plot_type="bar", feature_names=columns, show=False)
                plt.title(f"Estimator DT - SHAP Bar")
                plt.show()
        elif classifier_name == 'RF':
            explainer = shap.TreeExplainer(model.estimators_[1])
            shap_values = explainer.shap_values(data)
            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    print(f"Class {i} summary plot:")
                    shap.summary_plot(class_shap_values, data, max_display=10, show=False)
                    plt.title(f"Estimator RF - Class {i} SHAP Summary")
                    plt.show()
                    shap.summary_plot(class_shap_values, data, plot_type="bar", feature_names=columns, show=False)
                    plt.title(f"Estimator RF - Class {i} SHAP Bar")
                    plt.show()
            else:
                shap.summary_plot(shap_values, data, max_display=10, show=False)
                plt.title(f"Estimator RF - SHAP Summary")
                plt.show()
                shap.summary_plot(shap_values, data, plot_type="bar", feature_names=columns, show=False)
                plt.title(f"Estimator RF - SHAP Bar")
                plt.show()
        elif classifier_name == 'KNN':
            if not model.estimators_.contains('KNN'):
                print("KNN not found in estimators; You're probabily using Fast-ELENIDS.")
                return
            explainer = shap.KernelExplainer(model.estimators_[3].predict, data)
            shap_values = explainer.shap_values(data)
            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    print(f"Class {i} summary plot:")
                    shap.summary_plot(class_shap_values, data, max_display=10, show=False)
                    plt.title(f"Estimator KNN - Class {i} SHAP Summary")
                    plt.show()
                    shap.summary_plot(class_shap_values, data, plot_type="bar", feature_names=columns, show=False)
                    plt.title(f"Estimator KNN - Class {i} SHAP Bar")
                    plt.show()
            else:
                shap.summary_plot(shap_values, data, max_display=10, show=False)
                plt.title(f"Estimator KNN - SHAP Summary")
                plt.show()
                shap.summary_plot(shap_values, data, plot_type="bar", feature_names=columns, show=False)
                plt.title(f"Estimator KNN - SHAP Bar")
                plt.show()
        else:
            explainer = shap.Explainer(model.estimators_[2].predict, data)
            shap_values = explainer.shap_values(data)
            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    print(f"Class {i} summary plot:")
                    shap.summary_plot(class_shap_values, data, max_display=10, show=False)
                    plt.title(f"Estimator MLP - Class {i} SHAP Summary")
                    plt.show()
                    shap.summary_plot(class_shap_values, data, plot_type="bar", feature_names=columns, show=False)
                    plt.title(f"Estimator MLP - Class {i} SHAP Bar")
                    plt.show()
            else:
                shap.summary_plot(shap_values, data, max_display=10, show=False)
                plt.title(f"Estimator MLP - SHAP Summary")
                plt.show()
                shap.summary_plot(shap_values, data, plot_type="bar", feature_names=columns, show=False)
                plt.title(f"Estimator MLP - SHAP Bar")
                plt.show()



def feature_importance(model, columns):
    """
    Provide a graph of the relevance of the features for the model.

    Parameters
    ----------
    model : ELENIDS or F-ELENIDS.
    
    data : the columns of the UNSW-NB15 dataset (without the last two).

    Returns
    -------
    Info regarding the 
    """
    feature_importances = pd.DataFrame(model.estimators_[0].feature_importances_, index=columns, columns=["Importance"])
    feature_importances1 = pd.DataFrame(model.estimators_[1].feature_importances_, index=columns, columns=["Importance"])
    
    #print(feature_importances)

    for col in range(0, len(columns)):
        feature_importances["Importance"][col] = (feature_importances["Importance"][col] + feature_importances1["Importance"][col]) / 2

    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importances.plot(kind='bar', figsize=(8,6), title="Feature importance of the model")

def example():
    """
    Simulates an IDS that uses the package functions.
    """
    df = load()
    le = LabelEncoder()
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    df['proto']         = le.fit_transform(df['proto'])
    df['service']       = le1.fit_transform(df['service'])
    df['state']         = le2.fit_transform(df['state'])
    df['attack_cat']    = le3.fit_transform(df['attack_cat'])

    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[:-2]], df['attack_cat'], test_size=0.3)

    df['proto'] = le.inverse_transform(df['proto'])    
    df['service'] = le1.inverse_transform(df['service'])
    df['state'] = le2.inverse_transform(df['state'])
    dataframes = [X_test.iloc[:2000],X_test.iloc[2000:4000], X_test.iloc[4000:6000], X_test.iloc[6000:8000], X_test.iloc[8000:10000], X_test.iloc[10000:12000], X_test.iloc[12000:14000], X_test.iloc[14000:16000],X_test.iloc[16000:18000], X_test.iloc[18000:22000], X_test.iloc[22000:24000], X_test.iloc[24000:26000], X_test.iloc[26000:28000], X_test.iloc[28000:30000] ]
    #tests = [Y_test.iloc[:2000],Y_test.iloc[2000:4000], Y_test.iloc[4000:6000], Y_test.iloc[6000:8000], Y_test.iloc[8000:10000], Y_test.iloc[10000:12000], Y_test.iloc[12000:14000], Y_test.iloc[14000:16000],Y_test.iloc[16000:18000], Y_test.iloc[18000:22000], Y_test.iloc[22000:24000], Y_test.iloc[24000:26000], Y_test.iloc[26000:28000], Y_test.iloc[28000:30000] ]    


    ids = build()
    ids = fit(ids, X_train, Y_train)
    col = df.columns[ : -2]

    for dataframe in dataframes:
        classify(ids, dataframe, col, le, le1, le2, le3)



