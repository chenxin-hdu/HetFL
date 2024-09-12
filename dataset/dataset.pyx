import logging
import os
import _pickle as pickle
import random
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
except ImportError:
    from ..config import ProjectConfig


class DictCustomDataset(Dataset):
    def __init__(self, str root_dir, list selected_versions=[1], int embed_d=300, use_resample=True,
                 int item_len_limit=256, version_cache: dict = None):
        if version_cache is None:
            version_cache = dict()
        self.root_dir = root_dir
        self.selected_versions = [selected_versions] if isinstance(selected_versions, int) else selected_versions
        self.embed_d = embed_d
        self.use_resample = use_resample
        self.item_len_limit = item_len_limit
        self.dataset = list()
        self.dataset_dict = dict()
        self.version_cache = version_cache
        self.dataset_pass_test_not_none_index_list = list()
        self.dataset_fail_test_not_none_index_list = list()
        self.dataset_history_not_none_index_list = list()
        self.dataset_method_call_not_none_index_list = list()
        self.dataset_label_true = list()
        self.dataset_label_false = list()
        self.load()
        self.build()
        logging.info(f"Label True: {len(self.dataset_label_true)}")
        logging.info(f"Label False: {len(self.dataset_label_false)}")
        if use_resample:
            self.resample_for_balance_label()
            self.resample_for_failed_test()

    def load(self):
        self.dataset_dict = dict()
        self.dataset_dict["label"] = dict()
        self.dataset_dict["pass_test"] = dict()
        self.dataset_dict["fail_test"] = dict()
        self.dataset_dict["history"] = dict()
        self.dataset_dict["method_call"] = dict()
        self.dataset_dict["method"] = dict()
        cdef dict label = self.dataset_dict["label"]
        cdef dict pass_test = self.dataset_dict["pass_test"]
        cdef dict fail_test = self.dataset_dict["fail_test"]
        cdef dict history = self.dataset_dict["history"]
        cdef dict method_call = self.dataset_dict["method_call"]
        cdef dict method = self.dataset_dict["method"]
        logging.info(f"Loading...")
        for version in tqdm(self.selected_versions):
            version_path = f"{self.root_dir}/{version}"
            with open(f"{version_path}/{ProjectConfig.labels_filename}", "rb") as f:
                t = pickle.load(f)
                label.update(self.rename_key_with_versions(t, version))
            with open(f"{version_path}/{ProjectConfig.vector_pass_test_attribute_filename}", "rb") as f:
                t = pickle.load(f)
                pass_test.update(self.rename_key_with_versions(t, version))
            with open(f"{version_path}/{ProjectConfig.vector_fail_test_attribute_filename}", "rb") as f:
                t = pickle.load(f)
                fail_test.update(self.rename_key_with_versions(t, version))
            with open(f"{version_path}/{ProjectConfig.vector_method_history_filename}", "rb") as f:
                t = pickle.load(f)
                history.update(self.rename_key_with_versions(t, version))
            with open(f"{version_path}/{ProjectConfig.vector_method_call_filename}", "rb") as f:
                t = pickle.load(f)
                method_call.update(self.rename_key_with_versions(t, version))
            with open(f"{version_path}/{ProjectConfig.vector_method_attribute_filename}", "rb") as f:
                t = pickle.load(f)
                method.update(self.rename_key_with_versions(t, version))

    def build(self):
        self.dataset = list()
        cdef dict dataset_temp_label = self.dataset_dict["label"]
        cdef dict dataset_temp_pass_test = self.dataset_dict["pass_test"]
        cdef dict dataset_temp_fail_test = self.dataset_dict["fail_test"]
        cdef dict dataset_temp_history = self.dataset_dict["history"]
        cdef dict dataset_temp_method_call = self.dataset_dict["method_call"]
        cdef dict dataset_temp_method = self.dataset_dict["method"]
        del self.dataset_dict
        cdef list dataset_label_true_temp = self.dataset_label_true
        cdef list dataset_label_false_temp = self.dataset_label_false
        dataset_temp = self.dataset
        cdef list dataset_pass_test_not_none_index_list_temp = self.dataset_pass_test_not_none_index_list
        cdef list dataset_fail_test_not_none_index_list_temp = self.dataset_fail_test_not_none_index_list
        cdef list dataset_history_not_none_index_list_temp = self.dataset_history_not_none_index_list
        cdef list dataset_method_call_not_none_index_list_temp = self.dataset_method_call_not_none_index_list
        logging.info(f"Building...")
        __pass_test_zero_tensor = 2
        __fail_test_zero_tensor = 2
        __history_zero_tensor = 3
        __method_call_zero_tensor = 3
        cdef int i = 0
        cdef str key
        cdef int num_test = len(dataset_temp_pass_test) + len(dataset_temp_fail_test)
        cdef int num_method=len(dataset_temp_label)
        logging.info(f"Find {num_test} tests.")
        logging.info(f"Find {num_method} methods.")
        cdef int num_edges = 0
        for key, label in tqdm(dataset_temp_label.items()):
            method = dataset_temp_method.get(key)
            if method is None:
                continue
            method = [method]
            if label:
                dataset_label_true_temp.append(i)
            else:
                dataset_label_false_temp.append(i)
            pass_test = dataset_temp_pass_test.get(key)
            fail_test = dataset_temp_fail_test.get(key)
            history = dataset_temp_history.get(key)
            method_call = dataset_temp_method_call.get(key)
            if pass_test is None:
                pass_test = __pass_test_zero_tensor
            else:
                num_edges += len(pass_test)
            if fail_test is None:
                fail_test = __fail_test_zero_tensor
            else:
                num_edges += len(fail_test)
            if history is None:
                history = __history_zero_tensor
            # else:
            #     num_edges += len(history)
            if method_call is None:
                method_call = __method_call_zero_tensor
            # else:
            #     num_edges += len(method_call)
            dataset_temp.append((pass_test, fail_test, history, method_call, method, label))
            i += 1
        logging.info(f"Find {num_edges} edges.")

    def rename_key_with_versions(self, dict d, int version):
        cdef dict new_dict = dict()
        cdef str key
        for key, value in d.items():
            new_dict[f"{key}-{version}"] = value
        return new_dict

    def resample_for_balance_label(self):
        cdef int true_num = len(self.dataset_label_true)
        cdef int false_num = len(self.dataset_label_false)
        if true_num == 0:
            logging.warning(f"No bug method in dataset.")
            logging.info(f"Resampled for balancing label with 0 items.")
            return
        if false_num == 0:
            logging.info(f"Skip balancing label for 0 false label.")
            return
        cdef resample_num = false_num - true_num
        cdef count = 0
        if resample_num >= 0:
            while resample_num > 0:
                current_resample_num = min(resample_num, true_num)
                for i in self.dataset_label_true[:current_resample_num]:
                    self.dataset.append(self.dataset[i])
                resample_num -= current_resample_num
                count += current_resample_num
        else:
            resample_num = -resample_num
            while resample_num > 0:
                current_resample_num = min(resample_num, false_num)
                for i in self.dataset_label_false[:current_resample_num]:
                    self.dataset.append(self.dataset[i])
                resample_num -= current_resample_num
                count += current_resample_num
        logging.info(f"Resampled for balancing label with {count} items.")

    def resample_for_failed_test(self):
        # We have used a more convenient method here, which is slightly different from the one in the paper
        for i, t in enumerate(self.dataset):
            if isinstance(t, np.ndarray):
                self.dataset[i][1] = self.instant_resample_for_failed_test(t, length=len(self.dataset[i][0]))

    def instant_resample_for_failed_test(self, data: np.ndarray, length=None):
        if length is None:
            length = int(data.shape[0])
        if isinstance(data, torch.Tensor):
            return data
        else:
            data_length = len(data)
            data = np.array(data)
            return data[np.random.choice(length, size=length) % data_length][:random.randint(1, length)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, int idx):
        cdef list new_item
        cdef int item_len_limit = self.item_len_limit
        item = self.dataset[idx]
        new_item = list()
        for i, t in enumerate(item):
            if isinstance(t, torch.Tensor):
                new_item.append(t)
            elif isinstance(t, bool):
                new_item.append(torch.tensor(t, dtype=torch.float32))
            elif isinstance(t, int):
                new_item.append(torch.zeros((1, t, self.embed_d), dtype=torch.float32))
            else:
                if len(t) > item_len_limit:
                    t = t[:item_len_limit]
                new_item.append(torch.tensor(np.array(t), dtype=torch.float32))
        passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = new_item
        return passed_test_cases, failed_test_cases, history_change, call_information, methods, labels


cpdef fit_with_max(data_list):
    cdef int max_length = 0, t_length
    cdef list new_data
    for t in data_list:
        t_length = int(t.shape[0])
        if max_length < t_length:
            max_length = t_length
    new_data = list()
    for t in data_list:
        new_data.append(torch.cat([t, torch.zeros([max_length - t.shape[0], *t.shape[-2:]])]))
    return torch.stack(new_data, dim=0)

cpdef tuple dic_collate_fn(batch):
    passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = zip(*batch)
    passed_test_cases = fit_with_max(passed_test_cases)
    failed_test_cases = fit_with_max(failed_test_cases)
    history_change = fit_with_max(history_change)
    call_information = fit_with_max(call_information)
    return passed_test_cases, failed_test_cases, history_change, call_information, torch.stack(methods,
                                                                                               dim=0), torch.stack(
        labels, dim=0)

if __name__ == '__main__':
    dataset = DictCustomDataset("/cxgroup/zdl/dataset/defects4j-2.0/compiled/Closure")
    data = dataset[0]
    pass
