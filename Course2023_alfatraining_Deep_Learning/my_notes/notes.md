# NOTES #

notes and important links for alfatraining course in deep learning, in 2023  
(version: 08:00 26.01.2024)  


Deep Learning Book: A. Geron, 2023, "Praxiseinstieg Machine Learning mit Scikit-Learn, Keras und TensorFlow"  

* [Machine Learning Book web](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)  
* [Machine Learning Book](https://homl.info/er3)  
* [Codes from the book in github](https://github.com/ageron/handson-ml3)  


Neural Networks videos from 3Blue 1Brown:  

* [Neural Network youtube playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  


First day discussion links:  

* [TensorFlow and Differential Equations](https://medium.com/@fjpantunes2/tensorflow-and-differential-equations-a-simple-example-77d88d98ea3e), on medium  
* [Physics-informed neural networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks), on wikipedia  
* [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), on wikipedia  


Encoding:  

* [One-hot encoding](https://en.wikipedia.org/wiki/One-hot), on wikipedia  
* [1-aus-n-Code](https://de.wikipedia.org/wiki/1-aus-n-Code), on german wikipedia  


Rangfolge:  

* [Ranking](https://en.wikipedia.org/wiki/Ranking), on wikipedia  
* [Rangordnung](https://de.wikipedia.org/wiki/Rangordnung), on german wikipedia  


TensorFlow:  

* [TensorFlow website](https://www.tensorflow.org/)  
* [TensorFlow installation with Anaconda](https://docs.anaconda.com/free/anaconda/applications/tensorflow/)  
* [TensorFlow installation with pip](https://www.tensorflow.org/install/pip#windows-native)  
* [TensorFlow](https://en.wikipedia.org/wiki/TensorFlow), on wikipedia  
* [TensorFlow](https://de.wikipedia.org/wiki/TensorFlow), on german wikipedia  
* [TensorFlow](https://id.wikipedia.org/wiki/TensorFlow), di wikipedia Indonesia  


Keras:  

* [Keras website](https://keras.io/)  
* [Keras API](https://keras.io/api/)  
* [Keras](https://en.wikipedia.org/wiki/Keras), on wikipedia  
* [Keras](https://de.wikipedia.org/wiki/Keras), on german wikipedia  


PyTorch:

* [PyTorch website](https://pytorch.org/)  
* [PyTorch](https://en.wikipedia.org/wiki/PyTorch), on wikipedia  
* [PyTorch](https://de.wikipedia.org/wiki/PyTorch), on german wikipedia  
* [PyTorch](https://id.wikipedia.org/wiki/PyTorch), di wikipedia Indonesia  


OpenCV-Python

* [OpenCV-Python Tutorials](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html)  
* [opencv-python on pypi](https://pypi.org/project/opencv-python/) as wrapper package for OpenCV python bindings  


OpenNN:  open-source neural networks library for machine learning with C++  

* [OpenNN website](https://www.opennn.net/)  
* [OpenNN on github](https://github.com/Artelnics/opennn)  
* [OpenNN](https://en.wikipedia.org/wiki/OpenNN), on wikipedia  
* [OpenNN](https://de.wikipedia.org/wiki/OpenNN), on german wikipedia  


Apache Mahout:  distributed linear algebra framework and mathematically expressive Scala DSL  

* [Apache Mahout website](https://mahout.apache.org/)  
* [Apache Mahout](https://en.wikipedia.org/wiki/Apache_Mahout), on wikipedia  


Google JAX:  machine learning framework for transforming numerical functions  

* [Google JAX on github](https://github.com/google/jax)  
* [Google JAX](https://en.wikipedia.org/wiki/Google_JAX), on wikipedia  


MATLAB & Octave:  

* [MATLAB](https://www.mathworks.com/products/matlab.html)  
* [Octave](https://octave.org/)  


first week coding tips:  

* [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape)  
* ["cloning" a row or column vector to a matrix](https://stackoverflow.com/questions/1550130/cloning-row-or-column-vectors)  
* [Error in Python script "Expected 2D array, got 1D array instead:"](https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead)  
* [Reading tab-delimited file with Pandas](https://stackoverflow.com/questions/27896214/reading-tab-delimited-file-with-pandas-works-on-windows-but-not-on-mac)  
* [Load a pandas DataFrame, with TensorFlow](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe)  


### Project: EEG Data processing and classification (CANCELED)  

This topic were not chosen because the data loading was tricky and also my data was credential. 


Useful links for project:  

* [MNE in Python](https://github.com/mne-tools/mne-python), for MEG and EEG analysis and visualization  
* [MNE website](https://mne.tools/stable/index.html#)  
* [MNE dev](https://mne.tools/dev/index.html)  
* [MNE dev installation guide](https://mne.tools/dev/install/manual_install.html#manual-install)  
* [MNE installation guide](https://mne.tools/0.23/install/index.html)  
* [example from my MATLAB/EEGLAB code](https://bitbucket.org/iscab/bci_naovibe_footrace/src/master/EEG-analysis-script/scripts/ana01_filt_coaNAO.m)  


### Project: Image Super Resolution  

This topic is chosen because of data availability. 

Hints:  

* Super-Resolution  
* ResNet: Residual Network  
* Skip-connection  
* Batch Normalization  
* Peak signal-to-noise ratio (PSNR)  
* Structural similarity index measure (SSIM)  
* Image Quality metrics  
* Matplotlib  


drizzle algorithm:  

* suggestion "in astronomy one commonly uses a drizzle algorithm, perhaps you want to put that in as some layer?"  
* [Linear Reconstruction of the Hubble Deep Field](https://www.stsci.edu/~fruchter/dither/drizzle.html), explaining drizzle algorithm  
* suggestion "i suspect a working approach would be to apply drizzle 4x to enlarge the picture linearly and then apply a neuronal network, perhaps with convolutions in otder sharpen the image and make it smaller. the result is then a 2x larger image which is sharp"  


check:  

* [alfatraining](https://www.alfatraining.de/gefoerderte-weiterbildung/) courses  
* [alfatraining](https://www.alfatraining.de/) website  
* detailed in [my private repository](https://bitbucket.org/iscab/alfatraining_2023_deep_learning/)  
* [this notes on github](https://github.com/iscab/belajar_python/blob/main/Course2023_alfatraining_Deep_Learning/my_notes/notes.md)  
* [this notes on bitbucket](https://bitbucket.org/iscab/alfatraining_2023_deep_learning/src/master/my_notes/notes.md)  


version: 08:00 26.01.2024  

# End of File

