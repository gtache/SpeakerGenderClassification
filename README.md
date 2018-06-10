# Speaker gender classification
This assignment was to compare different classifiers for the gender recognition of speakers in audio files.    
The files were taken from the LibriSpeech dev-clean dataset.    
The features used are the MFCC features extracted using librosa.

## File descriptions
- GenderPredictions.ipynb is the notebook containing the report for this assignment
- Settings.py contains program-wide settings for the assignment
- Utils.py contains methods related to data processing
- main.py contains the main method as well as the method used to test a classifier
- results.txt contains notes of the result obtained during the development of the assignment. Note that due to changes in the program and the preprocessing of the data during the development, the results may be unreproducible. This is mainly there for "historical" purposes.

- classifier module
    - Classifier.py contains the base classifier class
    - NNClassifier.py contains the base neural network classifier class
    - ConstantClassifier.py contains the constant (always male) classifier
    - LinearClassifier.py contains the Linear SVC classifier
    - RFClassifier.py contains the Random Forest classifier
    - CNNClassifier.py contains the Convolutional Neural Network classifier
    - SNNClassifier.py contains the Shallow Neural Network classifier

- data folder
    - Features_with_label.npy is the dump of the numpy array containing the features along with their label (as an array of tuples)
    - Genders.txt is a simple SpeakerID to Gender label file
    - Max.npy is the dump of the maxima of the MFCC features as an array
    - Min.npy is the dump of the minima of the MFCC features as an array

- the models folder contains the dumps of the models tested