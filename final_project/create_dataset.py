import os
from cta import CTAs
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# precompute and physically store the MIP sequences for training and validation
def create_dataset():
    pos_dir = "../data/processed/exp_occlusion"
    anno_dir = "../data/processed/exp_occlusion_annotation"
    neg_dir = "../data/processed/exp_control"
    root = "/media/vince/FreeAgent Drive/cta_mip_seqs"
    excel = "../data/processed/train_val_data_annotations.xlsx"
    pos_sheet, neg_sheet = "occlusion", "control"
    pos_box_info, neg_box_info = (excel, pos_sheet), (excel, neg_sheet)
    format = 'nii'
    n_slices = 20
    n_neighbors = 5
    axis = 'z'

    pos_ctas = CTAs(pos_dir, anno_dir)
    neg_ctas = CTAs(neg_dir)
    pos_ctas.save_mip_seqs_dataset(
                dir=root,
                img_format=format,
                num_slices=n_slices,
                box_info=pos_box_info,
                axis=axis,
                num_neighbors=n_neighbors)
    neg_ctas.save_mip_seqs_dataset(
                dir=root,
                img_format=format,
                num_slices=n_slices,
                box_info=neg_box_info,
                axis=axis,
                num_neighbors=n_neighbors)

# precompute and physically store the MIP sequences for testing
def create_testset():
    pos_dir = "../data/source/patient_wise_test_data/positive"
    neg_dir = "../data/source/patient_wise_test_data/negative"
    root = "/media/vince/FreeAgent Drive/test_seqs"
    excel = "../data/processed/test_data_annotations.xlsx"
    pos_sheet, neg_sheet = "occlusion", "control"
    pos_box_info, neg_box_info = (excel, pos_sheet), (excel, neg_sheet)
    format = 'nii'
    n_slices = 20
    n_neighbors = 5
    axis = 'z'

    #pos_ctas = CTAs(pos_dir)
    neg_ctas = CTAs(neg_dir)
    '''
    pos_ctas.save_mip_seqs_testset(
                dir=root,
                img_format=format,
                num_slices=n_slices,
                box_info=pos_box_info,
                axis=axis,
                num_neighbors=n_neighbors,
                pos=True)
    '''
    neg_ctas.save_mip_seqs_testset(
                dir=root,
                img_format=format,
                num_slices=n_slices,
                box_info=neg_box_info,
                axis=axis,
                num_neighbors=n_neighbors,
                pos=False)

# train/validation split for precomputed MIP sequences
def train_val_split(root, pos_dir, neg_dir, test_ratio, balanced=False, random_seed=0):
    pos_seqs_dir = os.path.join(root, pos_dir)
    neg_seqs_dir = os.path.join(root, neg_dir)
    seqs_dataset = dict()
    for seq_dir in os.listdir(pos_seqs_dir):
        seq_path = os.path.join(pos_seqs_dir, seq_dir)
        seqs_dataset[seq_path] = 1
    for seq_dir in os.listdir(neg_seqs_dir):
        seq_path = os.path.join(neg_seqs_dir, seq_dir)
        seqs_dataset[seq_path] = 0
    x = [seq_path for seq_path in seqs_dataset.keys()]
    y = [seq_label for seq_label in seqs_dataset.values()]
    if balanced:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        for train_indices, val_indices in sss.split(x, y):
            train_data = {x[train_index] : y[train_index] for train_index in train_indices}
            val_data = {x[val_index] : y[val_index] for val_index in val_indices}
    else:
        ss = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        for train_indices, val_indices in ss.split(x):
            train_data = {x[train_index] : y[train_index] for train_index in train_indices}
            val_data = {x[val_index] : y[val_index] for val_index in val_indices}
    return train_data, val_data

# compute pos/neg MIP sequences counts for train/validation sets
def train_val_counts(train_data, val_data):
    total_pos_seqs = 0
    total_neg_seqs = 0
    train_pos_seqs = 0
    train_neg_seqs = 0
    val_pos_seqs = 0
    val_neg_seqs = 0
    data = {**train_data, **val_data}
    for seq_label in data.values():
        if seq_label == 0:
            total_neg_seqs += 1
        elif seq_label == 1:
            total_pos_seqs += 1
    for seq_label in train_data.values():
        if seq_label == 0:
            train_neg_seqs += 1
        elif seq_label == 1:
            train_pos_seqs += 1
    for seq_label in val_data.values():
        if seq_label == 0:
            val_neg_seqs += 1
        elif seq_label == 1:
            val_pos_seqs += 1
    return total_pos_seqs, total_neg_seqs, train_pos_seqs, train_neg_seqs, val_pos_seqs, val_neg_seqs


if __name__ == "__main__":
    #create_dataset()
    create_testset()
