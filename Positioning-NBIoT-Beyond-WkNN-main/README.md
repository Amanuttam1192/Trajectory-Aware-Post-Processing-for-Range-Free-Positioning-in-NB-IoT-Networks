# Range-free Positioning in NB-IoT Networks by Machine Learning: beyond WkNN
Positioning in NB-IoT networks by fingerprinting using machine learning

This Matlab code performs positioning in NB-IoT networks by fingerprinting using five machine learning strategies based on combinations of Weighted k Nearest Neighbours (WkNN), Support Vector Machine (SVM), Random Forest (RF) and Artificial Neural Networks (ANN) and of data preprocessing techniques aimed at reducing computational
complexity in the spatial domain and in the feature domain, respectively by means of hard ans soft clustering and of Principal Component Analysis (PCA). The five strategies are defined as follows:

I - WkWN in combination with the Weighted Coverage similarity metric, combining a Euclidean distance computed on the selected 3GPP radio parameter (RSSI, SINR, RSRP, RSRQ) and the number of NBIoT Physical Cell Identifier (NPCI) in common between points;

II - Hard clustering (k-Means or k-Medoids) + SVM + WkWN using Weighted Coverage;

III - Hard clustering (k-Means or k-Medoids) + RF + WkWN using Weighted Coverage;

IV - Soft clustering (Fuzzy C-Means) + RF + WkWN using Weighted Coverage;

V - ANN.


Each strategy is implemented in a main script and a set of supporting functions duplicated over two different datasets:
- the 'Oslo' dataset, containing data for seven measurement campaigns carried out in 2019 in Oslo, Norway;
- the 'Rome' dataset, containing data for six campaigns carried out in 2021 in Rome, Italy.

Please note that due to space limitations, the repository contains the data for each dataset already preprocessed and stored in two Matlab workspace files. The data in raw form are available in the following open source dataset:

L. De Nardis, G. Caso, Ö. Alay, M. Neri, A. Brunstrom, M.-G. Di Benedetto, Outdoor NB-IoT and 5G coverage and channel information data in urban environments - https://zenodo.org/record/7674298

also including 5G data.

When using this code in a scientific publication, please cite the following research paper:

L. De Nardis, M. Savelli, G. Caso, F. Ferretti, L. Tonelli, N. Bouzar, A. Brunstrom, Ö. Alay, M. Neri, F. Elbahhar and M.-G. Di Benedetto,  “Range-free Positioning in NB-IoT Networks by Machine Learning: Beyond WkNN”, IEEE Journal on Indoor and Seamless Positioning and Navigation, 2025. DOI: 10.1109/JISPIN.2025.3558465

Acknowledgments

The work of Luca De Nardis and Nadir Bouzar in the creation of this dataset was partially supported by the European Union - Next Generation EU under the Italian National Recovery and Resilience Plan (NRRP), Mission 4, Component 2, Investment 1.3, CUP B53C22004050001, partnership on “Telecommunications of the Future” (PE00000001 - program “RESTART”).
