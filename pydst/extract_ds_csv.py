"""A function that extracts the data from the folders
    and converts the mp3 files to numpy arrays.
    
This function can be used as both a standalone function
or imported in a different class.
"""

import io
import os
import csv
import logging
import numpy as np
from array import array
from pydst import DEFAULT_SEED
from pydub import AudioSegment
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
    # try:
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
    # except:
    #     err_str = "File: {0} missing.".format(filename)
    #     logger.error(err_str)
    #     raise ValueError(err_str)

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


def extract_data(mp3_filenames, trackid, targets, root):
    """Extract mp3 files and convert to numpy arrays.
        This data is saved in npy files in tracks folder.
        
    :param mp3_filenames
    :param trackid
    :param root
    """
    error_files = []
    for idx, mp3_filename in enumerate(mp3_filenames):
        load_filename = root + 'mp3_files/' + mp3_filename
        save_filename = root + 'csv_files/' + str(trackid[idx]) + '.csv'
        try:
            song = AudioSegment.from_mp3(load_filename).get_array_of_samples().tolist()
        except:
            error_files.append({'mp3': mp3_filename, 'tid': trackid[idx], 'idx': idx})
        else:
            song.extend(targets[idx].tolist())
            with open(save_filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(song)


def seperate_merge(tids, split):
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
    size_of_dataset = tids.shape[0]
    test_sz, valid_sz, train_sz = int(round(size_of_dataset * split[2])), int(round(size_of_dataset * split[1])), int(round(
        size_of_dataset * split[0]))

    train_targets = tids[0:train_sz]
    valid_targets = tids[train_sz:train_sz + valid_sz]
    test_targets = tids[train_sz + valid_sz: -1]

    return train_targets, valid_targets, test_targets


def get_dataset(rng, root_folder, data_div, _size_of):
    """Function to perform the functions to extract the data.
    
    :param rng 
    :param root_folder
    :param _size_of: Number of samples.
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
    extract_data(mp3_files, tids, targets, root_folder)
    logger.info("Data extracted from mp3 files and saved")

    # Check if 3 divisions are given (train, valid, test)
    if len(data_div) != 3:
        err_str = "The data can be divided in 3. {} divisions given!".format(len(data_div))
        logger.error(err_str)
        raise ValueError(err_str)

    # Seperate in test, valid, training sets - 20, 10, 70
    [train_targets, valid_targets, test_targets] = seperate_merge(tids, data_div)
    logger.info("Data separated and merged into dictionaries")

    return train_targets, valid_targets, test_targets, map_of_labels


def save_archives(train_md, valid_md, test_md, label_map, root):
    """Function to save the data in archives.
    
    :param train_md 
    :param valid_md 
    :param test_md 
    :param label_map
    """
    for idx in range(3):
        if idx == 0:
            savename = root + 'train_files.txt'
            set = train_md

        elif idx == 1:
            savename = root + 'valid_files.txt'
            set = valid_md
        else:
            savename = root + 'test_files.txt'
            set = test_md
        with open(savename, 'w') as file:
            for item in set:
                content = str(item) + '.csv'
                file.write('{}\n'.format(content))

    with open('label_map.txt', 'w') as file:
        for label in label_map:
            content = str(label)
            file.write('{}\n'.format(content))


if __name__ == "__main__":
    # Extract the dataset
    cwd = os.getcwd()
    dataset_folder = cwd+'/magnatagatune/'
    rndState = np.random.RandomState(DEFAULT_SEED)
    size_of_sets = 10
    down_sampling = 1
    divisions = [0.7, 0.1, 0.2]
    [train_targets, valid_targets, test_targets, label_map] = get_dataset(rndState,
                                                                          dataset_folder,
                                                                          divisions,
                                                                          size_of_sets)
    logger.info("Extracted the metadata and save npy files")

    # Save data into archive
    save_archives(train_targets, valid_targets, test_targets, label_map, dataset_folder)
    logger.info("Saved the different sets of metadata")
