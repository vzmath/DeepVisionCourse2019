import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# single patient CTA
###################################################################################################################
class CTA:
    def __init__(self, src):
        self.img = nib.load(src)
        self.path = os.path.abspath(src)
    # basic methods for CTA information
    def get_shape(self):
        return self.img.shape
    def get_filepath(self):
        return self.path
    def get_filename(self):
        return self.path.split('/')[-1]
    def get_patient_id(self):
        return self.get_filename().split('.')[0]
    def get_affine(self):
        return self.img.affine

    # methods for computing single MIP and single MIP sequence
    def load_head_box(self, excel_path, sheet):
        record_df = pd.read_excel(excel_path, sheet_name=sheet, index_col='patient id')
        patient_id = int(self.get_patient_id())
        box_stats = record_df.loc[patient_id]
        coords = ['x start', 'x end', 'y start', 'y end', 'z start', 'z end']
        box = {coord : idx for coord, idx in zip(coords, box_stats)}
        return box
    def rand_slice_num(self, num_slices=20):
        lower = int(num_slices * (1 - 0.2))
        upper = int(num_slices * (1 - 0.1))
        lower_range = list(range(lower, upper+1))
        lower = int(num_slices * (1 + 0.1))
        upper = int(num_slices * (1 + 0.2))
        upper_range = list(range(lower, upper+1))
        total_range = set(lower_range) | set(upper_range)
        total_range.add(int(num_slices))
        total_range = np.array(list(total_range))
        return np.random.choice(total_range), total_range
    def single_mip(self, start_slice, end_slice, axis='x', box=None):
        if box is None:
            if axis.lower() == 'x':
                voi = self.img.dataobj[start_slice:end_slice+1, :, :]
                mip = np.amax(voi, axis=0)
            elif axis.lower() == 'y':
                voi = self.img.dataobj[:, start_slice:end_slice+1, :]
                mip = np.amax(voi, axis=1)
            elif axis.lower() == 'z':
                voi = self.img.dataobj[:, :, start_slice:end_slice+1]
                mip = np.amax(voi, axis=2)
            else:
                print("Axis {} is not a valid axis for MIPs computations, exiting...".format(axis))
                sys.exit(1)
        else:
            brain = tuple( coord for coord in box.values() )
            x_start, x_end, y_start, y_end, z_start, z_end = brain
            if axis.lower() == 'x':
                voi = self.img.dataobj[start_slice:end_slice+1, y_start:y_end, z_start:z_end]
                mip = np.amax(voi, axis=0)
            elif axis.lower() == 'y':
                voi = self.img.dataobj[x_start:x_end, start_slice:end_slice+1, z_start:z_end]
                mip = np.amax(voi, axis=1)
            elif axis.lower() == 'z':
                voi = self.img.dataobj[x_start:x_end, y_start:y_end, start_slice:end_slice+1]
                mip = np.amax(voi, axis=2)
            else:
                print('Axis {} is not a valid axis for MIPs computation, exiting...'.format(axis))
                sys.exit(1)
        return mip
    def mip_seqs(self, start_slice, end_slice, axis='x', box=None, num_neighbors=1):
        slice_num = end_slice - start_slice + 1     # number of slices used to compute a single MIP
        index_range = np.arange(start_slice-num_neighbors, start_slice+num_neighbors+1) # left-end start index and right-end start index
        mip_seq = [ self.single_mip(index, index+slice_num-1, axis, box) for index in index_range ]  # store MIP seqs in a list
        return mip_seq, (start_slice, end_slice)
    def get_occl_slices(self, anno_obj, axis='x'):
        occl_slices = list()
        if axis.lower() == 'x':
            idx_range, _, _ = self.get_shape()
            for idx in np.arange(idx_range):    # don't use np.arange() func, otherwise the dataobj slicing won't work properly
                src_slice = self.img.get_fdata()[idx, :, :]
                anno_slice = anno_obj.img.get_fdata()[idx, :, :]
                elem_dot_prod = np.multiply(src_slice, anno_slice)  # the element-wise product should be nonzero if the underlying slice is part of the occlusion bounding box
                if np.count_nonzero(elem_dot_prod) != 0:
                    occl_slices.append(idx)
        elif axis.lower() == 'y':
            _, idx_range, _ = self.get_shape()
            for idx in range(idx_range):
                src_slice = self.img.get_fdata()[:, idx, :]
                anno_slice = anno_obj.img.get_fdata()[:, idx, :]
                elem_dot_prod = np.multiply(src_slice, anno_slice)
                if np.count_nonzero(elem_dot_prod) != 0:
                    occl_slices.append(idx)
        elif axis.lower() == 'z':
            _, _, idx_range = self.get_shape()
            for idx in range(idx_range):
                src_slice = self.img.get_fdata()[:, :, idx]
                anno_slice = anno_obj.img.get_fdata()[:, :, idx]
                elem_dot_prod = np.multiply(src_slice, anno_slice)
                if np.count_nonzero(elem_dot_prod) != 0:
                    occl_slices.append(idx)
        else:
            print("Error, please check the axis parameter or file dimension matches...")
            sys.exit(1)
        return np.array(occl_slices)

    # methods for computing patient-wise MIPs and MIP sequences (train/validation sets, or test sets)
    def mip_seqs_dataset(self, num_slices=20, box=None, axis='x', num_neighbors=1, anno_obj=None):
        mip_seqs_data = dict()
        # return bounding box for brain area if there exists one, otherwise use the entire source CTA
        if box is None:
            x_start, y_start, z_start = 0, 0, 0
            x_end, y_end, z_end = self.get_shape()
        else:
            brain = tuple( coord for coord in box.values() )
            x_start, x_end, y_start, y_end, z_start, z_end = brain
        # start/end slices for the specified axis
        if axis.lower() == 'x':
            start_slice, end_slice = x_start, x_end
        elif axis.lower() == 'y':
            start_slice, end_slice = y_start, y_end
        elif axis.lower() == 'z':
            start_slice, end_slice = z_start, z_end
        else:
            print("Axis {} is not a valid axis, please choose from x, y, or z and try again...".format(axis))
            sys.exit(1)
        # compute MIP sequences
        patient_id = self.get_patient_id()
        print("Start computing MIP sequences for patient {}".format(patient_id))
        if anno_obj is not None:
            occl_slices = self.get_occl_slices(anno_obj, axis)
        for current_start_slice in tqdm(np.arange(start_slice, end_slice)):
            current_num_slices, _ = self.rand_slice_num(num_slices)
            current_end_slice = current_start_slice + current_num_slices - 1
            if current_start_slice - num_neighbors >= start_slice and current_end_slice + num_neighbors <= end_slice:   # ensure that each MIP will have proper number of neighbors
                # documenting mip sequence information
                mip_seq, seq_idx_info = self.mip_seqs(current_start_slice, current_end_slice, axis, box, num_neighbors)
                seq_start_idx, seq_end_idx = seq_idx_info
                mip_seq_info = (patient_id, seq_start_idx, seq_end_idx)

                if anno_obj is not None:
                    mip_seq_slice_range = np.array( list(range(current_start_slice-num_neighbors, current_end_slice-num_neighbors+1)) )
                    shared_slices = np.intersect1d(occl_slices, mip_seq_slice_range)
                    # handle labeling for positive patients
                    if shared_slices.size == 0:
                        mip_seqs_data[mip_seq_info] = (mip_seq, 0)
                    else:
                        mip_seqs_data[mip_seq_info] = (mip_seq, 1)
                else:   # default labeling for negative patients
                    mip_seqs_data[mip_seq_info] = (mip_seq, 0)
        print("MIP sequences of patient {} have been successfully computed".format(patient_id))
        return mip_seqs_data
    # methods for computing patient-wise MIP sequences for test sets
    def mip_seqs_testset(self, num_slices=20, box=None, axis='x', num_neighbors=1):
        mip_seqs_data = dict()
        # return bounding box for brain area if there exists one, otherwise use the entire source CTA
        if box is None:
            x_start, y_start, z_start = 0, 0, 0
            x_end, y_end, z_end = self.get_shape()
        else:
            brain = tuple( coord for coord in box.values() )
            x_start, x_end, y_start, y_end, z_start, z_end = brain
        # start/end slices for the specified axis
        if axis.lower() == 'x':
            start_slice, end_slice = x_start, x_end
        elif axis.lower() == 'y':
            start_slice, end_slice = y_start, y_end
        elif axis.lower() == 'z':
            start_slice, end_slice = z_start, z_end
        else:
            print("Axis {} is not a valid axis, please choose from x, y, or z and try again...".format(axis))
            sys.exit(1)
        # compute MIP sequences without labeling
        patient_id = self.get_patient_id()
        print("Start computing MIP sequences for patient {}".format(patient_id))
        for current_start_slice in tqdm(np.arange(start_slice, end_slice)):
            current_num_slices, _ = self.rand_slice_num(num_slices)
            current_end_slice = current_start_slice + current_num_slices - 1
            if current_start_slice - num_neighbors >= start_slice and current_end_slice + num_neighbors <= end_slice:
                mip_seq, seq_idx_info = self.mip_seqs(current_start_slice, current_end_slice, axis, box, num_neighbors)
                seq_start_idx, seq_end_idx = seq_idx_info
                mip_seq_info = (patient_id, seq_start_idx, seq_end_idx)
                mip_seqs_data[mip_seq_info] = mip_seq
        print("MIP sequences of patient {} have been successfully computed".format(patient_id))
        return mip_seqs_data
###################################################################################################################

# directory of CTAs, i.e., positive/negative patients
###################################################################################################################
class CTAs:
    def __init__(self, src_dir, anno_dir=None):
        self.src_path = src_dir
        self.anno_path = anno_dir
        self.cta_paths = [os.path.join(self.src_path, file) for file in os.listdir(self.src_path)]
        self.cta_objs = [CTA(cta_path) for cta_path in self.cta_paths]
    # generate train/validation MIP sequences on the fly
    def mip_seqs_dataset(self, num_slices=20, box_info=None, axis='x', num_neighbors=1):
        mip_seq_data = dict()
        for cta in tqdm(self.cta_objs):
            patient_id = cta.get_patient_id()
            if box_info is None:
                box = None
            else:
                excel, sheet = box_info
                box = cta.load_head_box(excel, sheet)
            if self.anno_path is not None:
                anno_file = "{}_mask.nii.gz".format(patient_id)
                anno_obj_path = os.path.join(self.anno_path, anno_file)
                anno_obj = CTA(anno_obj_path)
                current_cta_mip_seq_data = cta.mip_seqs_dataset(num_slices, box, axis, num_neighbors, anno_obj)
            else:
                current_cta_mip_seq_data = cta.mip_seqs_dataset(num_slices, box, axis, num_neighbors)
            for mip_seq_info, (mip_seq, seq_label) in current_cta_mip_seq_data.items():
                mip_seq_data[mip_seq_info] = (mip_seq, seq_label)
        return mip_seq_data
    # generate test MIP sequences on the fly
    def mip_seqs_testset(self, num_slices=20, box_info=None, axis='x', num_neighbors=1):
        mip_seq_data = dict()
        for cta in tqdm(self.cta_objs):
            if box_info is None:
                box = None
            else:
                excel, sheet = box_info
                box = cta.load_head_box(excel, sheet)
            current_cta_mip_seq_data = cta.mip_seqs_testset(num_slices, box, axis, num_neighbors)
            for mip_seq_info, mip_seq in current_cta_mip_seq_data.items():
                mip_seq_data[mip_seq_info] = mip_seq
        return mip_seq_data

    # save train/validation MIP sequences physically
    def save_mip_seqs_dataset(self, dir='.', img_format='nii', num_slices=20, box_info=None, axis='x', num_neighbors=1):
        occl_seqs_dir = os.path.join(dir, "occlusion")
        cont_seqs_dir = os.path.join(dir, "control")
        if not os.path.exists(occl_seqs_dir):
            os.mkdir(occl_seqs_dir)
        if not os.path.exists(cont_seqs_dir):
            os.mkdir(cont_seqs_dir)

        for cta in tqdm(self.cta_objs):
            patient_id = cta.get_patient_id()
            affine = cta.get_affine()
            if box_info is None:
                box = None
            else:
                excel, sheet = box_info
                box = cta.load_head_box(excel, sheet)
            if self.anno_path is not None:
                anno_file = "{}_mask.nii.gz".format(patient_id)
                anno_obj_path = os.path.join(self.anno_path, anno_file)
                anno_obj = CTA(anno_obj_path)
                current_cta_mip_seq_data = cta.mip_seqs_dataset(num_slices, box, axis, num_neighbors, anno_obj)
            else:
                current_cta_mip_seq_data = cta.mip_seqs_dataset(num_slices, box, axis, num_neighbors)
            print("Start saving MIP sequences for patient {}".format(patient_id))
            for mip_seq_info, (mip_seq, seq_label) in tqdm(current_cta_mip_seq_data.items()):
                seq_idx = 1
                _, seq_start, seq_end = mip_seq_info
                seq_dirname = "{}_{}_{}_seqs".format(patient_id, seq_start, seq_end)
                if seq_label == 0:
                    seq_dir = os.path.join(cont_seqs_dir, seq_dirname)
                elif seq_label == 1:
                    seq_dir = os.path.join(occl_seqs_dir, seq_dirname)
                else:
                    print("{} is not a valid MIP sequence label...".format(seq_label))
                # create the dir for the sequence data
                if not os.path.exists(seq_dir):
                    os.mkdir(seq_dir)

                for mip in mip_seq:
                    mip_filename = "{}_{}_{}_MIP_{}.{}".format(patient_id, seq_start, seq_end, seq_idx, img_format)
                    mip_path = os.path.join(seq_dir, mip_filename)
                    mip_img = nib.Nifti1Image(mip, affine)
                    nib.save(mip_img, mip_path)
                    seq_idx += 1
            print("MIP sequences of patient {} have been successfully saved".format(patient_id))
    # save test MIP sequences physically
    def save_mip_seqs_testset(self, dir='.', img_format='nii', num_slices=20, box_info=None, axis='x', num_neighbors=1, pos=False):
        occl_seqs_dir = os.path.join(dir, "occlusion")
        cont_seqs_dir = os.path.join(dir, "control")
        if not os.path.exists(occl_seqs_dir):
            os.mkdir(occl_seqs_dir)
        if not os.path.exists(cont_seqs_dir):
            os.mkdir(cont_seqs_dir)
        for cta in tqdm(self.cta_objs):
            patient_id = cta.get_patient_id()
            affine = cta.get_affine()
            if box_info is None:
                box = None
            else:
                excel, sheet = box_info
                box = cta.load_head_box(excel, sheet)
            current_cta_mip_seq_data = cta.mip_seqs_testset(num_slices, box, axis, num_neighbors)
            if pos:
                patient_dir = os.path.join(occl_seqs_dir, patient_id)
            else:
                patient_dir = os.path.join(cont_seqs_dir, patient_id)
            if not os.path.exists(patient_dir):
                os.mkdir(patient_dir)
            print("Start saving MIP sequences for patient {}".format(patient_id))
            for mip_seq_info, mip_seq in tqdm(current_cta_mip_seq_data.items()):
                seq_idx = 1
                _, seq_start, seq_end = mip_seq_info
                seq_dirname = "{}_{}_{}_seqs".format(patient_id, seq_start, seq_end)
                seq_dir = os.path.join(patient_dir, seq_dirname)
                if not os.path.exists(seq_dir):
                    os.mkdir(seq_dir)
                # save each MIP in one single MIP sequence
                for mip in mip_seq:
                    mip_filename = "{}_{}_{}_MIP_{}.{}".format(patient_id, seq_start, seq_end, seq_idx, img_format)
                    mip_path = os.path.join(seq_dir, mip_filename)
                    mip_img = nib.Nifti1Image(mip, affine)
                    nib.save(mip_img, mip_path)
                    seq_idx += 1
            print("MIP sequences of patient {} have been successfully saved".format(patient_id))
###################################################################################################################

# mixture of positive/negative directories of CTAs
###################################################################################################################
class MIP_Seq_Dataset:
    def __init__(self, *cta_dir_mip_seqs, test=False):
        self.test = test
        if test:
            self.mip_seq_dataset = {mip_seq_info : mip_seqs
                                    for cta_mip_seqs in cta_dir_mip_seqs
                                        for mip_seq_info, mip_seqs in cta_mip_seqs.items()}
        else:
            self.mip_seq_dataset = {mip_seq_info : (mip_seqs, seqs_label)
                                    for cta_mip_seqs in cta_dir_mip_seqs
                                        for mip_seq_info, (mip_seqs, seqs_label) in cta_mip_seqs.items()}

    def train_val_split(self, test_ratio, balanced=False, random_seed=0):
        if self.test:
            print("Current dataset only consists of MIP sequences for test, not valid for train/validation split...")
            sys.exit(1)
        else:
            x = [(seq_info, seqs) for seq_info, (seqs, _) in self.mip_seq_dataset.items()]
            y = [seq_label for _, seq_label in self.mip_seq_dataset.values()]
            if balanced:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
                for train_indices, test_indices in sss.split(x, y):
                    train_data = {x[train_index][0] : (x[train_index][1], y[train_index])
                                  for train_index in train_indices}
                    test_data = {x[test_index][0] : (x[test_index][1], y[test_index])
                                  for test_index in test_indices}
            else:
                ss = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
                for train_indices, test_indices in ss.split(x):
                    train_data = {x[train_index][0] : (x[train_index][1], y[train_index])
                                  for train_index in train_indices}
                    test_data = {x[test_index][0] : (x[test_index][1], y[test_index])
                                  for test_index in test_indices}
            return train_data, test_data
###################################################################################################################

if __name__ == "__main__":
    # some unit tests
    sample = '../data/processed/control'
    anno = '../data/processed/occlusion_annotation'
    excel = '../data/processed/train_val_data_annotations.xlsx'
    sheet = 'control'
    box_info = (excel, sheet)

    dataset = CTA_Dataset(sample)
    occlusion = dataset.mip_seqs_dataset(20, box_info, 'y', 5)
    print(len(occlusion))
