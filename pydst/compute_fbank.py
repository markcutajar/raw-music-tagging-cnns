# -*- coding: utf-8 -*-
"""Function to calculate fbank features of audio files in a certain location."""


from python_speech_features import logfbank
import numpy as np


def extract_fbanks(LOAD_LOCATION, SAVE_LOCATION, which_set, sample_rate=16000, fft_size=512, filters=26):
    
    data_folder = 'data1/'
    
    data_dir = LOAD_LOCATION + data_folder + which_set + '/'
    metad_path = root + data_folder + which_set + '_metadata.npy'
    
    assert os.path.isfile(self.metad_path), (
        'Metadata file does not exist at expected path: ' + self.metad_path)
    assert os.path.isdir(self.data_dir), (
        'Data directory does not exist at expected path: ' + self.data_dir)
    
    
    metad = np.load(self.metad_path).item()
    tids = metad['tids']
    
    for tid in tids:
        
        filename = self.data_dir + str(tid) + '.npy'
        file = np.load(self.data_dir + str(tid) + '.npy')
        
        fbank_file = logfbank(signal=file, sample_rate=sample_rate, nfft=fft_size, nfilt=filters, highfreq=None)
        
        savename = SAVE_LOCATION + data_folder + str(tid) + '.npy'
        np.save(savename, fbank_file)
        
        
if __name__ == "__main__":

    LOAD_LOCATION = 'magnatagatune/tracks/'
    SAVE_LOCATION = 'magnatagatune/features/'
    
    sets = ['train', 'valid', 'test']
    
    for setname in sets:
        extract_fbank(LOAD_LOCATION, SAVE_LOCATION, setname, 32000, 512, 40)
        