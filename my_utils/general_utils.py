import os
import shutil
import logging
import numpy as np


def mkdirs(*dirs_paths, erase=False):
    for path in dirs_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if erase:
                shutil.rmtree(path)
                os.makedirs(path)


def get_logger(save_log_path, overwrite=False):
    filemode = 'w' if overwrite else 'a'  
    formatter = '%(message)s : %(asctime)s'
    logging.basicConfig(level=logging.DEBUG, filename=save_log_path, filemode=filemode, format=formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    
    return logger


def args2string(args):
    string = '\n' + '-'*90 + '\n'
    for arg, value in vars(args).items():
        string += '{} : {}\n'.format(str(arg), value)
    string += '-'*90

    return string


def get_sorted_ids(first_list, second_list=None):
    if second_list is None:
        sorted_ids = np.argsort(first_list)
    else:
        sorted_ids = np.lexsort((second_list, first_list))
    sorted_ids = sorted_ids[::-1]
    return sorted_ids


def sort_by_length(first_key_list, second_key_list=None, other_apply_lists=None):
    num_others = 0
    second_sorted = None
    other_sorted = None

    # --- Pre-shuffle ---
    first_array = np.array(first_key_list)
    shuffled_ids = np.random.permutation(len(first_array))
    first_array = first_array[shuffled_ids]
    if second_key_list is not None:
        second_array = np.array(second_key_list)
        second_array = second_array[shuffled_ids]
    if other_apply_lists is not None:
        if isinstance(other_apply_lists, tuple):
            other_apply_lists = list(other_apply_lists)
            num_others = len(other_apply_lists)
            assert num_others > 1
            for i in range(num_others):
                if other_apply_lists[i] is not None:
                    other_apply_lists[i] = np.array(other_apply_lists[i])[shuffled_ids]
        else:
            num_others = 1
            other_apply_lists = np.array(other_apply_lists)[shuffled_ids]

    # --- Sort by length ---
    first_lens = [len(x) for x in first_array]
    if second_key_list is None:
        sorted_ids = get_sorted_ids(first_lens)
    else:
        second_lens = [len(x) for x in second_array]
        sorted_ids = get_sorted_ids(first_lens, second_lens)
        second_sorted = list(second_array[sorted_ids])
    first_sorted = list(first_array[sorted_ids])
    if other_apply_lists is not None:
        if num_others > 1:
            for i in range(num_others):
                if other_apply_lists[i] is not None:
                    other_apply_lists[i] = list(other_apply_lists[i][sorted_ids])
            other_sorted = tuple(other_apply_lists)
        else:
            assert num_others == 1
            other_sorted = list(other_apply_lists[sorted_ids])

    return first_sorted, second_sorted, other_sorted
