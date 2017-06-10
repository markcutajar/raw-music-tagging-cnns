# -*- coding: utf-8 -*-
"""Function to calculate fbank features of audio files in a certain location."""


from python_speech_features import logfbank
import numpy as np
import logging
import os
from time import gmtime, strftime

# Define logger, formatter and handler
LOGGER_FORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
TIME = strftime("%Y%m%d_%H%M%S", gmtime())
LOG_FILENAME = 'logs/fbe_'+TIME+'.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOGGER_FORMAT)
file_handler = logging.FileHandler(LOG_FILENAME)
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def extract_fbanks(LOAD_LOCATION, SAVE_LOCATION, which_set, samplerate=16000, fft_size=512, filters=26):
    
    logger.info("FBank extraction started!")
    data_dir = LOAD_LOCATION + 'tracks/'
    metad_path = LOAD_LOCATION + which_set + '_metadata.npz'
    
    assert os.path.isfile(metad_path), (
        'Metadata file does not exist at expected path: ' + metad_path)
    assert os.path.isdir(data_dir), (
        'Data directory does not exist at expected path: ' + data_dir)
    
    
    metad = np.load(metad_path)
    tids = metad['tids']
    
    logger.info("Iterating through files")
    for tid in tids:
        
        filename = data_dir + str(tid) + '.npy'
        file = np.load(data_dir + str(tid) + '.npy')
        
        fbank_file = logfbank(signal=file, samplerate=samplerate, nfft=fft_size, nfilt=filters, highfreq=None)
        
        savename = SAVE_LOCATION + 'tracks/' + str(tid) + '.npy'
        np.save(savename, fbank_file)
    np.savez(SAVE_LOCATION + which_set + '_metadata.npz', label_map=metad['label_map'],
             mp3_files=metad['mp3_files'], targets=metad['targets'], tids=metad['tids'])    
    logger.info("Files saved")   
        
if __name__ == "__main__":

    LOAD_LOCATION = 'magnatagatune/dataset/data1/'
    SAVE_LOCATION = 'magnatagatune/dataset/fbankfeatures/'
    
    sets = ['train', 'valid', 'test']
    
    for setname in sets:
        logger.info("Processing {} dataset".format(setname)) 
        extract_fbanks(LOAD_LOCATION, SAVE_LOCATION, setname, 16000, 512, 40)
        