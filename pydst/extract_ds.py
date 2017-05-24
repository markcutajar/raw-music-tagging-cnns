import csv
import logging
import numpy as np
from pydst import DEFAULT_SEED
from pydub import AudioSegment

# Define logger
LOGGER_FORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
LOG_FILENAME = 'cre_ds.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOGGER_FORMAT)
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def extract_tags_names(root):
    # Define lists to store data information
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
        logger.error("File: {0} missing.".format(filename))

    targets = np.asarray(targets)
    tids = np.asarray(tids)
    mp3_files = np.asarray(mp3_files)
    return label_map, mp3_files, targets, tids


def extract_data(mp3_filenames, root):
    data = []
    # Extract mp3 files and convert to numpy arrays
    for mp3_filename in mp3_filenames:
        filename =  root + "mp3_files/" + mp3_filename
        song = AudioSegment.from_mp3(filename).get_array_of_samples().tolist()
        data.append(song)
    data = np.asarray(data)

    # Confirm the sizes are as they need to be
    if len(mp3_filenames) != data.shape[0]:
        err_str = "mp3_filenames {} size not equal to extracted data size {}!".format(len(mp3_filenames), data.shape[0])
        logger.error(err_str)
        raise ValueError(err_str)
    return data


def shuffle(rng, targets, mp3_files, tids):
    perm = rng.permutation(targets.shape[0])
    ptargets = targets[perm]
    pmp3f = mp3_files[perm]
    ptids = tids[perm]
    return ptargets, pmp3f, ptids


def seperate(data, targets, mp3_files, tids, split):
    size_of_dataset = targets.shape[0]
    test_sz, valid_sz, train_sz = round(size_of_dataset * split[2]), \
                                  round(size_of_dataset * split[1]), \
                                  round(size_of_dataset * split[0])

    trn_tgts, trn_data, trn_mp3f, trn_tids = targets[0:train_sz], \
                                             data[0:train_sz, :], \
                                             mp3_files[0:train_sz], \
                                             tids[0:train_sz]

    vld_tgts, vld_data, vld_mp3f, vld_tids = targets[train_sz:train_sz + valid_sz], \
                                             data[train_sz:train_sz + valid_sz, :], \
                                             mp3_files[train_sz:train_sz + valid_sz], \
                                             tids[train_sz:train_sz + valid_sz]


    tst_tgts, tst_data, tst_mp3f, tst_tids = targets[train_sz + valid_sz: -1], \
                                             data[train_sz + valid_sz: -1, :], \
                                             mp3_files[train_sz + valid_sz: -1], \
                                             tids[train_sz + valid_sz: -1]

    training_data = {'targets': trn_tgts, 'data': trn_data, 'mp3f': trn_mp3f, 'tids': trn_tids}
    validation_data = {'targets': vld_tgts, 'data': vld_data, 'mp3f': vld_mp3f, 'tids': vld_tids}
    test_data = {'targets': tst_tgts, 'data': tst_data, 'mp3f': tst_mp3f, 'tids': tst_tids}
    return training_data, validation_data, test_data


def get_dataset(rng, root, divisions, _size_of = -1):
    # Extract tags and names
    [label_map, mp3_files, targets, tids] = extract_tags_names(root)
    logger.info("Extracted tags and names into arrays")
    logger.info("Label_map {}, mp3_files {}, targets {}, tids {}".format(len(label_map), len(mp3_files),
                                                                         targets.shape, tids.shape))

    # Sort all targets to be sorted according to frequency
    # TODO: Create function to sort by frequency

    # Size reduction if needed
    mp3_files = mp3_files[0:_size_of]
    targets = targets[0:_size_of]
    tids = tids[0:_size_of]

    # Shuffle names and targets
    [targets, mp3_files, tids] = shuffle(rng, targets, mp3_files, tids)
    logger.info("Shuffled targets, mp3 files and tids")

    # Extract data from mp3 files
    data = extract_data(mp3_files, root)
    logger.info("Data extracted from mp3 files")

    if len(divisions) != 3:
        err_str = "The data can be divided in 3. {} divisions given!".format(len(divisions))
        logger.error(err_str)
        raise ValueError(err_str)

    # Seperate in test, valid, training sets - 20, 10, 70
    [trndata, vlddata, tstdata] = seperate(data, targets, mp3_files, tids, divisions)
    return trndata, vlddata, tstdata, label_map


def save_archives(trndata,vlddata, tstdata, label_map):
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
    size_of_sets = 5
    divisions = [0.7, 0.1, 0.2]
    [trndata, vlddata, tstdata, label_map] = get_dataset(rng, root, divisions, _size_of=size_of_sets)
    logger.info("Extracted the dataset")

    # Save data into archive
    save_archives(trndata, vlddata, tstdata, label_map)
    logger.info("Saved the different sets of data")
