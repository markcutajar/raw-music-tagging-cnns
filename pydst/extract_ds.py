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
import pydub
from time import gmtime, strftime

# Define logger, formatter and handler
LOGGER_FORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
TIME = strftime("%Y%m%d_%H%M%S", gmtime())
LOG_FILENAME = 'logs/ext_ds_'+TIME+'.log'

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
    """Extract the targets, mp3_files, tids and label_map
    
    :param root: the root folder where the files of the
        magnatagatune dataset is found.
    :returns: targets, mp3_files, tids, targets_to_labels 
    """
    targets, mp3_files, tids, targets_to_labels = [], [], [], []

    # Open file and store the information in the appropriate lists
    filename = root + 'annotations_final.csv'
    try:
        with open(filename, newline='') as f:
                annotations_file = csv.reader(f, delimiter='\t')

                for idx, row in enumerate(annotations_file):
                    if idx == 0:
                        targets_to_labels = row
                        del(targets_to_labels[0], targets_to_labels[-1])
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
    targets_to_labels = np.asarray(targets_to_labels)
    return targets, mp3_files, tids, targets_to_labels


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


def extract_data(mp3_filenames, trackid, root, down_sample_factor):
    """Extract mp3 files and convert to numpy arrays.
        This data is saved in npy files in tracks folder.
        
    :param mp3_filenames
    :param trackid
    :param root
    :param down_sample_factor
    """

    error_files = []
    for idx, mp3_filename in enumerate(mp3_filenames):
        load_filename = root + "mp3_files/" + mp3_filename
        save_filename = root + "tracks/" + str(trackid[idx]) + ".npy"
        try:
            song = AudioSegment.from_mp3(load_filename).get_array_of_samples().tolist()
        except:
            error_files.append({'mp3': mp3_filename, 'tid': trackid[idx], 'idx': idx})
        else:
            song = np.asarray(song[::down_sample_factor])
            np.save(save_filename, song)
    np.save(root + 'error_files.npy', error_files)


def seperate_merge(targets, mp3_files, tids, split):
    """Function to seperate the data according to the 
        splits defined.
    
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

    trn_tgts, trn_mp3f, trn_tids = targets[0:train_sz], mp3_files[0:train_sz], tids[0:train_sz]

    vld_tgts, vld_mp3f, vld_tids = targets[train_sz:train_sz + valid_sz], mp3_files[train_sz:train_sz + valid_sz], tids[
                                                                                        train_sz:train_sz + valid_sz]

    tst_tgts, tst_mp3f, tst_tids = targets[train_sz + valid_sz: -1], mp3_files[train_sz + valid_sz: -1], tids[train_sz + valid_sz: -1]

    training_data = {'targets': trn_tgts, 'mp3f': trn_mp3f, 'tids': trn_tids}
    validation_data = {'targets': vld_tgts, 'mp3f': vld_mp3f, 'tids': vld_tids}
    test_data = {'targets': tst_tgts, 'mp3f': tst_mp3f, 'tids': tst_tids}
    return training_data, validation_data, test_data


def get_dataset(rng, root_folder, data_div, _size_of=-1, _down_sampling_window=3):
    """Function to perform the functions to extract the data.
    
    :param rng 
    :param root_folder 
    :param data_div 
    :param _size_of: Number of samples.
    :param _down_sampling_window: Down sampling factor.
    :returns: trn_data, vld_data, tst_data, label_map
    """

    # Extract tags and names
    [targets, mp3_files, tids, map_of_labels] = extract_tags_names(root_folder)
    logger.info("Extracted tags and names into arrays")
    logger.info("Label_map {}, mp3_files {}, targets {}, tids {}".format(len(map_of_labels), len(mp3_files),
                                                                         targets.shape, tids.shape))
    # Shuffle names and targets
    [targets, mp3_files, tids] = shuffle(rng, targets, mp3_files, tids)
    logger.info("Shuffled targets, mp3 files and tids")

    # Sample reduction if needed
    [targets, mp3_files, tids] = reduction_samples(targets, mp3_files, tids, _size_of)
    logger.info("Number of samples reduced")
    
    # Sort all targets to be sorted according to frequency
    [targets, map_of_labels] = sort_tags(targets, map_of_labels)
    logger.info("Tags sorted according to frequency")

    # Extract data from mp3 files and save npy
    extract_data(mp3_files, tids, root_folder, down_sample_factor=_down_sampling_window)
    logger.info("Data extracted from mp3 files and saved")

    # Check if 3 divisions are given (train, valid, test)
    if len(data_div) != 3:
        err_str = "The data can be divided in 3. {} divisions given!".format(len(data_div))
        logger.error(err_str)
        raise ValueError(err_str)

    # Seperate in test, valid, training sets - 20, 10, 70
    [train, valid, test] = seperate_merge(targets, mp3_files, tids, data_div)
    logger.info("Data separated and merged into dictionaries")

    return train, valid, test, map_of_labels


def save_archives(train_md, valid_md, test_md, lmap):
    """Function to save the data in archives.
    
    :param train_md 
    :param valid_md 
    :param test_md 
    :param lmap
    """
    train_archive, valid_archive, test_archive = "mtat_metatrain.npz", "mtat_metavalid.npz", "mtat_metatest.npz"
    np.savez(train_archive, label_map=lmap, mp3_files=train_md['mp3f'], targets=train_md['targets'],
             tids=train_md['tids'])

    np.savez(valid_archive, label_map=lmap, mp3_files=valid_md['mp3f'], targets=valid_md['targets'],
             tids=valid_md['tids'])

    np.savez(test_archive, label_map=lmap, mp3_files=test_md['mp3f'], targets=test_md['targets'],
             tids=test_md['tids'])


if __name__ == "__main__":
    # Extract the dataset
    dataset_folder = "magnatagatune/"
    rndState = np.random.RandomState(DEFAULT_SEED)
    size_of_sets = -1
    down_sampling = 1
    divisions = [0.7, 0.1, 0.2]
    [trndata, vlddata, tstdata, label_map] = get_dataset(rndState, dataset_folder, divisions, _size_of=size_of_sets,
                                                         _down_sampling_window=down_sampling)
    logger.info("Extracted the metadata and save npy files")

    # Save data into archive
    save_archives(trndata, vlddata, tstdata, label_map)
    logger.info("Saved the different sets of metadata")
