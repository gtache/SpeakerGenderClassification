# Speaker gender classification
This assignment was to compare different classifiers for the gender recognition of speakers in audio files.    
The files were taken from the LibriSpeech dev-clean dataset.    
The features used are the MFCC features extracted using librosa.

## Libraries used
- Keras 2.1.6
- tensorflow(-gpu) 1.8.0 with cuDNN 7.0 and CUDA 9.0
- scikit-learn 0.19.1
- numpy 1.14.3
- librosa 0.6.1 (needed ffmpeg bin on Path on Windows)

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
    - Features_with_label.npy is the dump of the numpy array containing the features along with their label (as an array of tuples) (can't be pushed to github due to its size) - Is available here https://drive.google.com/file/d/1y_GgtTNHm5SlOLw4pH29R4EsToymf3wr/view?usp=sharing
    - Genders.txt is a simple SpeakerID to Gender label file
    - Max.npy is the dump of the maxima of the MFCC features as an array
    - Min.npy is the dump of the minima of the MFCC features as an array

- the models folder contains the dumps of the models tested
