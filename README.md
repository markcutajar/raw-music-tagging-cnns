# Automatic Music Tagging using Convolutional Neural Networks

This repository contains all the code, documentation, and logs of the project regarding using convolutional nueral networks to automatically tag music. This is an MSc dissertation project at the University of Edinbrugh, partly also subsidised by Malta's Endeavour scholarship scheme 2016-2017.

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

The main dataset considered for this project is the MagnaTagATune dataset. This dataset was presented by Law et al. [1]. MagnaTagATune has 25,863 tracks from 230 different artists and 188 tags. The audio is available in mp3 format. Since only 230 different artists are present in this dataset, it lacks variety; however, it is used extensively and a good option to use to compare to other literature. This dataset is criticized to have error is the annotated tags [2], and the tags availability is unbalanced where the most common tag has __XX__ observations whilst the least common has __XX__. In the unearthed literature, the top 50 tags are used where the 50th tag has __XX__ observations. Moreover, there is a structural issue with the tags where even the top 50 tags have tags such as female, woman and female voice present among others. This would make it more difficult for the network to train for voice related tags. An option would be to merge these tags together; however, once again this would make the results not comparable to other literature.<br>

An alternative was the Million song dataset presented by Bertin-Mahieux et al. [3]. This has much more variety with more than three million songs. Since the dataset is so large, this can be very big and hence would take a considerable amount of time to process. For the time scale of this project, this is not an ideal dataset. Tests would take longer to train and validate, and would require more memory. An option would be to create a balanced subset from this dataset, however, that would still require to acquire the dataset which ranges in Terabytes and process it to acquire a balanced set. Once again this was deemed unfeasible.<br><br>

The final alternative was the newly setup FMA dataset [4]. This dataset has about 106,574 tracks from 16,341 different artists whilst also having audio available. This has more variety in the artists and the songs; the variety could aid the network generalize better. However, in this case specific tags apart from the genre, era and artist name are not available. Attempts to acquire the corresponding tags from the Last.FM API were mostly unsuccessful. The script designed, for each song would send an API request to Last.FM for the tags of the specific song. However, a large number of songs were not tagged by the general public. Moreover, since this dataset is relatively new and presented just in 2017, comparison to literature would be limited.<br>

Due to the issues with the latter two datasets, the MagnaTagATune dataset was deemed most appropriate, both due to the manageable size, tag availability and comparability to previous literature.<br>

[1] E. Law, K. West, M. Mandel, M. Bay, and J. S. Downie, “Evaluation of Algorithms using Games: The case of Music Tagging,” in ISMIR, 2009.<br>
[2]	K. Choi, G. Fazekas, M. Sandler, and K. Cho, “Convolutional Recurrent Neural Networks for Music Classification,” arXiv, 2016, url: https://arxiv.org/pdf/1609.04243.pdf.<br>
[3]	T. Beretin-Mahieux, D. P. W. Ellis, P. Lamere, and B. Whitman, “The Million Song Dataset,” in 12th International Society for Music Information Retrieval Conference, 2011.<br>
[4]	M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, “FMA: A Dataset for Music Analysis,” url: https://arxiv.org/pdf/1612.01840.pdf.<br>

### Framework

This project will make use of Tensorflow Python API v1.1 and Cloud SDK. The models are first tested on the local environment and then exported to be trained on Cloud ML. The models are trained in distributed manner in order to minimize the computational time and hence larger networks can be used. 

As much as possible the low level API shall be used with some minor exceptions. This is done so the graph can be customised as much as needed for the different tests.
<br><br>
*To be continued...*


