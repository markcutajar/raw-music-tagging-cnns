# Automatic Music Tagging using Convolutional Neural Networks

This repository contains all the code, documentation, and logs of the project regarding using convolutional nueral networks to automatically tag music. This is a dissertation project at the University of Edinbrugh, partly also subsidised by Malta's Endeavour scholarship scheme 2016-2017.

Author: Mark Cutajar <br>
Email: mcut94@gmail.com / markc@gummymelon.com

Superviser: Professor Steve Renals <br>
School of Informatics <br>
University of Edinburgh

Package dependencies:
1. Python 2.7 (or 3.5, however Cloud SDK requires python 2.7 to run)
2. Tensorflow
3. Cloud SDK
4. pydub
5. ffmpeg
6. python_speech_features

Package with all the functions called pydst.
<br><br>
### Research proposal abstract

Online databases have grown to a size that are unmanageable without automatic tagging systems. This has incited interest in the field of automatic music classification. Recently, there has been a shift towards music tagging models making use of convolutional networks working upon raw waveforms. Even though raw waveforms have more information, in recent studies the spectrogram state-of-the-art is still not surpassed by any of the raw waveform models. This study aims to construct a number of the best performing raw-waveform models such as exclusively convolutional and convolutional recurrent models, and analyse the different feature maps. These feature maps will be investigated by listening to the extracted audio and comparing the audio to the ones of other models. Furthermore, extraction of the frequency spectrum of the different maps enables the construction of the filters of each stage by comparison to the previous level. Secondary objectives include construction of deeper networks, and making use of techniques such as residual learning and stochastic pooling. Thus, the secondary objective aims to find a model which is better than the state-of-the-art. The qualitative analysis of this study would not only aid researchers when constructing models for music classification and time-series classification, but also developers in the commercial scenario in order to construct models for music database management and recommendation systems. 
<br><br>
### Datasets considered

The main dataset conisdered for this project is the magnatagatune dataset. This dataset was presented by Edith Law, Kris West, Michael Mandel, Mert Bay and J. Stephen Downie. Evaluation of algorithms using games: the case of music annotation. In Proceedings of the 10th International Conference on Music Information Retrieval (ISMIR 2009), 2009.
Magnatagatune has 25,863 tracks from 230 different artists. In this dataset the audio is available in mp3 format. It can be seen that this dataset does not have a large variety of artists whilst being relatively small. However, it is simple ot use as the songs are pretagged with 188 different annotations.

An alternative is the Million song dataset presented by Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011. This has much more variety but the data can range upto around 100TB which for the time scale of this study is a bit too much.

Another alternative is the newly setup FMA dataset. This dataset presented in April 2017 by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson. *FMA: A Dataset for Music Analysis. In arXiv 2017. This dataset has about 106,574 tracks from 16,341 different artists whilst also having audio available. However, in this case specific tags apart from the genre, era and artist name are not available. This was seen as a better dataset from the Magnatagatune Dataset. However, attempts to acquire the tags from the Last.FM API where mostly unsuccessful due a large number of songs without user tags.

The FMA dataset might be used further on in the study to compare the effectivness of the models.
<br><br> 
*To be continued...*


