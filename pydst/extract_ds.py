"""A function that extracts the data from the folders
    and converts the mp3 files to numpy arrays.
    
This function can be used as both a standalone function
or imported in a different class.
"""

import csv
import logging
import numpy as np
from pydst import DEFAULT_SEED
from pydub import AudioSegment
from time import gmtime, strftime

# TODO: DRAW HOW DATA IS FLOWING THOUGHT THE FUNCTIONS
# TODO: CREATE A MOCK DATASET AND TEST IE CREATE A TEST DRIVEN FUNCTION

# Define logger, formatter and handler
LOGGER_FORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
TIME = strftime("%Y%m%d_%H%M%S", gmtime())
LOG_FILENAME = 'cre_ds_'+TIME+'.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOGGER_FORMAT)
file_handler = logging.FileHandler(LOG_FILENAME)
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def extract_tags_names(root):
    """Extract the targets, mp3_fules, tids and label_map
    
    :param root: the root folder where the files of the
        magnatagatune dataset is found.
    :returns: targets, mp3_files, tids, label_map 
    """
    targets, mp3_files, tids, label_map = [], [], [], []

    # Open file and store the information in the appropriate lists
    filename = root + 'annotations_final.csv'
    try:
        with open(filename, newline='') as f:
                annotations_file = csv.reader(f, delimiter='\t')

                for idx, row in enumerate(annotations_file):
                    if idx == 0:
                        label_map = row
                        del(label_map[0], label_map[-1])
                    else:

                        tids.append(row[0])
                        mp3_files.append(row[-1])
                        targets.append([float(i) for i in row[1:-1]])
    except:
        err_str = "File: {0} missing.".format(filename)
        logger.error(err_str)
        raise ValueError(err_str)

    targets = np.asarray(targets)
    tids = np.asarray(tids)
    mp3_files = np.asarray(mp3_files)
    label_map = np.asarray(label_map)
    return label_map, mp3_files, targets, tids


def extract_data(mp3_filenames, root, down_sample_factor):
    """Find each mp3_file from the folder and converts
        the mp3 file into numpy array.
    
    :param mp3_filenames: The name of the mp3 files to 
        be extracted.
    :param root: The folder where the magnatagatune 
        dataset is found.
    :return: data: Extracted mp3 files in numpy arrays.    
    """
    data = []
    # Extract mp3 files and convert to numpy arrays
    for idx, mp3_filename in enumerate(mp3_filenames):
        filename =  root + "mp3_files/" + mp3_filename
        song = AudioSegment.from_mp3(filename).get_array_of_samples().tolist()
        data.append(song)
        song = song[::down_sample_factor]
        np.save(root + "tracks/" + str(idx) + ".npy", song)
    data = np.asarray(data)

    # Confirm the sizes are as they need to be
    if len(mp3_filenames) != data.shape[0]:
        err_str = "mp3_filenames {} size not equal to extracted data size {}!".format(len(mp3_filenames), data.shape[0])
        logger.error(err_str)
        raise ValueError(err_str)


    return data


def sort_tags(targets, labels):
    """Function to sort the targets and labels according
        to frequency.
        
    :param targets: The targets of the labels to be sorted. 
    :param labels: The labels of the targets.
    :return: target: Sorted targets and labels.
    :return: labels: Sorted labels according to frequency
        of the targets.
    """
    sum_of_targets = np.sum(targets, axis=0)
    indices = np.flipud(np.argsort(sum_of_targets))
    sorted_targets = targets[:, indices]
    sorted_labels = labels[indices]
    return sorted_targets, sorted_labels


def reduction_samples(targets, mp3_files, tids, keep_rows):
    """Reduction of the number of files to be searched.
    
    :param targets
    :param mp3_files
    :param tids
    :param keep_rows 
    :returns targets, mp3_files, tids, keep_rows
    """
    mp3_files = mp3_files[0:keep_rows]
    targets = targets[0:keep_rows]
    tids = tids[0:keep_rows]
    return targets, mp3_files, tids


def shuffle(rng, targets, mp3_files, tids):
    """Shuffle of the targets to be found according to 
        rng.
    
    :param rng: Random generator class.
    :param targets
    :param mp3_files 
    :param tids
    :returns targets, mp3_files, tids
    """
    perm = rng.permutation(targets.shape[0])
    ptargets = targets[perm]
    pmp3f = mp3_files[perm]
    ptids = tids[perm]
    return ptargets, pmp3f, ptids


def seperate(data, targets, mp3_files, tids, split):
    """Function to seperate the data according to the 
        splits defined.
    
    :param data
    :param targets 
    :param mp3_files 
    :param tids 
    :param split: Split array with the fractions of
        training, validation and test set sizes.
    :return training_data: Dictionary with the training
        data.
    :return validation_data: Dictionary with the valid
        data.
    :return test_data: Dictionary with the test data.
    """
    size_of_dataset = targets.shape[0]
    test_sz, valid_sz, train_sz = round(size_of_dataset * split[2]), round(size_of_dataset * split[1]), round(
        size_of_dataset * split[0])

    trn_tgts, trn_data, trn_mp3f, trn_tids = targets[0:train_sz], data[0:train_sz, :], mp3_files[0:train_sz], tids[0:train_sz]

    vld_tgts, vld_data, vld_mp3f, vld_tids = targets[train_sz:train_sz + valid_sz], data[train_sz:train_sz + valid_sz, :], \
                                             mp3_files[train_sz:train_sz + valid_sz], tids[train_sz:train_sz + valid_sz]

    tst_tgts, tst_data, tst_mp3f, tst_tids = targets[train_sz + valid_sz: -1], data[train_sz + valid_sz: -1, :], \
                                             mp3_files[train_sz + valid_sz: -1], tids[train_sz + valid_sz: -1]

    training_data = {'targets': trn_tgts, 'data': trn_data, 'mp3f': trn_mp3f, 'tids': trn_tids}
    validation_data = {'targets': vld_tgts, 'data': vld_data, 'mp3f': vld_mp3f, 'tids': vld_tids}
    test_data = {'targets': tst_tgts, 'data': tst_data, 'mp3f': tst_mp3f, 'tids': tst_tids}
    return training_data, validation_data, test_data


def get_dataset(rng, root, divisions, _size_of = -1):
    """Function to perform the functions to extract the data.
    
    :param rng 
    :param root 
    :param divisions 
    :param _size_of: Number of samples.
    :returns: trn_data, vld_data, tst_data, label_map
    """
    # Extract tags and names
    [label_map, mp3_files, targets, tids] = extract_tags_names(root)
    logger.info("Extracted tags and names into arrays")
    logger.info("Label_map {}, mp3_files {}, targets {}, tids {}".format(len(label_map), len(mp3_files), targets.shape, tids.shape))
        
    # Sample reduction if needed
    [targets, mp3_files, tids] = reduction_samples(targets, mp3_files, tids, _size_of)
    logger.info("Number of samples reduced")
    
    # Sort all targets to be sorted according to frequency
    [targets, label_map] = sort_tags(targets, label_map)
    logger.info("Tags sorted according to frequency")
    
    # Shuffle names and targets
    [targets, mp3_files, tids] = shuffle(rng, targets, mp3_files, tids)
    logger.info("Shuffled targets, mp3 files and tids")

    # Extract data from mp3 files
    data = extract_data(mp3_files, root, down_sample_factor=3)
    logger.info("Data extracted from mp3 files")

    if len(divisions) != 3:
        err_str = "The data can be divided in 3. {} divisions given!".format(len(divisions))
        logger.error(err_str)
        raise ValueError(err_str)

    # Seperate in test, valid, training sets - 20, 10, 70
    [trndata, vlddata, tstdata] = seperate(data, targets, mp3_files, tids, divisions)
    return trndata, vlddata, tstdata, label_map


def save_archives(trndata,vlddata, tstdata, label_map):
    """Function to save the data in archives.
    
    :param trndata 
    :param vlddata 
    :param tstdata 
    :param label_map
    """
    train_archive, valid_archive, test_archive = "mtat_train.npz", "mtat_valid.npz", "mtat_test.npz"

    np.savez(train_archive, label_map=label_map, mp3_files=trndata['mp3f'], targets=trndata['targets'],
             tids=trndata['tids'], inputs=trndata['data'])

    np.savez(valid_archive, label_map=label_map, mp3_files=vlddata['mp3f'], targets=vlddata['targets'],
             tids=vlddata['tids'], inputs=vlddata['data'])

    np.savez(test_archive, label_map=label_map, mp3_files=tstdata['mp3f'], targets=tstdata['targets'],
             tids=tstdata['tids'], inputs=tstdata['data'])


if __name__ == "__main__":
    # Extract the dataset
    root = "magnatagatune/"
    rng = np.random.RandomState(DEFAULT_SEED)
    size_of_sets = 10
    divisions = [0.7, 0.1, 0.2]
    [trndata, vlddata, tstdata, label_map] = get_dataset(rng, root, divisions, _size_of=size_of_sets)
    logger.info("Extracted the dataset")

    # Save data into archive
    save_archives(trndata, vlddata, tstdata, label_map)
    logger.info("Saved the different sets of data")
