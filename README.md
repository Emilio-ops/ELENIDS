# Data-mining-for-cyber-security
This is the official directory for the Data mining for cyber security project.
The proposed system, Elenids, is an easy to use Passive NIDS. It provide a function to train the model and one to classify the wanted instances (passed as a npyarray that should satisfy the same requisites of the UNSW-NB15 dataset). The result is provided as an excel file called Report {timestamp}.
For any question, please open an issue or write to our mails.

To use the ids classification alghoritm you should before dawnload the following libraries:
1. Panda;
2. Sklearn;
3. Numpy.

To visualize the ids' statistics the following libraries are required:
1. matplotlib.pyplot;
2. Time.

To download the dataset (it is mandatory), please refer to this link: https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files%2FTraining%20and%20Testing%20Sets&ga=1

The ids classification alghoritm does not require anything else. It produces several excel files.
A complete implementation should also collect the packets from the network and pass them to the classification algorithm, but this is ouside the scope of our research.

To execute the code, please clone the repository and run from your preferred IDE or download the repository and run the code with "python ids_classification_algorithm.py" after entering the correct directory.

Please keep in mind that running the file from the terminal usually takes double the time than running the file from an IDE.

## Python package
pip install -i https://test.pypi.org/simple/ elenids to download the current distribution.
The package contains an example() function that simulate the whole process.
The adjourned package can be found here: https://test.pypi.org/project/elenids/ .
