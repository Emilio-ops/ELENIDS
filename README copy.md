# Elenids
This is a network-based, anomaly-based, passive intrusion detection system.
The system uses an ensemble of classifiers to classify netowork traffic.
Elenids is based on the UNSW-NB15 dataset and all the istances collected should be preprocessed using the same process.
The system offers two classifiers:
1. build(): produces a votingClassifier that uses four classification algorithms: DT, RF, KNN, MLP. 
            It's slower thand Fastbuild() but slightly more reliable.
2. Fastbuild(): produces a votingClassifier that uses three classification algorithms: DT, RF, MLP. 
            It's faster thand Fastbuild() but slightly less reliable.

The system produces an excel file for every batch passed to it, identified via timestamp.