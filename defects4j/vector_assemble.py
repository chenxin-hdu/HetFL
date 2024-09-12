import _pickle
import os
import sys
import time

from gensim.models import Word2Vec
from networkx import Graph
from tokenizers import Tokenizer
import numpy as np
import concurrent.futures
import logging
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
    from defects4j.fast_utils import preprocess_sentences
except ImportError:
    from ..config import ProjectConfig
    from .utils import Utils
    from .fast_utils import preprocess_sentences

num_sub_worker = 64
model_path = f"{ProjectConfig.path_dataset_home}/{ProjectConfig.word2vec}/{ProjectConfig.word2vec_model}"
model2_path = f"{ProjectConfig.path_dataset_home}/{ProjectConfig.word2vec}/{ProjectConfig.node2vec_model}"
logging.info(f"Load word2vec model.")
model = Word2Vec.load(model_path)
logging.info(f"Load node2vec model.")
model2 = Word2Vec.load(model2_path)

def process_word2vec(data: list, model: Word2Vec, tokenizer: Tokenizer, use_mean=True):
    vector = model.wv[
        [token for token in tokenizer.encode(preprocess_sentences(data)).tokens if token in model.wv]]
    if use_mean:
        vector = vector.mean(axis=0)
    return vector


def process_node2vec(data: Graph, model: Word2Vec, use_mean=True, embed_d=300):
    nodes = []
    edges = data.edges
    for item in edges:
        nodes.append(item[0]), nodes.append(item[1])
    if len(nodes) > 0:
        vector = np.stack([model.wv[node] for node in nodes if node in model.wv], axis=0)
        if use_mean:
            vector = vector.mean(axis=0)
        return vector
    else:
        return np.zeros((embed_d,))


def sub_task(data, version, main_id, sub_id, process_func, *process_param):
    try:
        # logging.info(f"Tasks-{sub_id} starts.")
        result = list()
        for d in data:
            if isinstance(d, list) or isinstance(d, tuple):
                t_result = list()
                for t in d:
                    t_result.append(process_func(t, *process_param))
                result.append(t_result)
            else:
                result.append(process_func(d, *process_param))
        Utils.create_path(f"{ProjectConfig.tmp_path}")
        with open(f"{ProjectConfig.tmp_path}/{version}_{main_id}_{sub_id}", "wb+") as f:
            _pickle.dump(result, f)
        # logging.info(f"Tasks-{sub_id} finished.")
    except:
        logging.error(traceback.format_exc())


def sub_task_method_call(data, version, main_id, sub_id, process_func, *process_param):
    try:
        # logging.info(f"Tasks-{sub_id} starts.")
        result = list()
        for d in data:
            if isinstance(d, list) or isinstance(d, tuple):
                t_result = list()
                for t in d:
                    tt_result = list()
                    tt_result.append(process_word2vec(t[0], process_param[0], process_param[2]))
                    tt_result.append(process_word2vec(t[1], process_param[0], process_param[2]))
                    tt_result.append(process_node2vec(t[2], process_param[1]))
                    t_result.append(tt_result)
                result.append(t_result)
        Utils.create_path(f"{ProjectConfig.tmp_path}")
        with open(f"{ProjectConfig.tmp_path}/{version}_{main_id}_{sub_id}", "wb+") as f:
            _pickle.dump(result, f)
        # logging.info(f"Tasks-{sub_id} finished.")
    except:
        logging.error(traceback.format_exc())


def batch_vector(data: list, version, process_func, *process_param, sub_task_func=sub_task):
    part_length = len(data) // num_sub_worker
    # logging.info(f"Distribute tasks...")
    main_id = time.time()
    preprocess_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=num_sub_worker)
    for i in range(num_sub_worker):
        if i == num_sub_worker - 1:
            part = data[i * part_length:]
        else:
            part = data[i * part_length:(i + 1) * part_length]
        preprocess_process_pool.submit(sub_task_func, part, version, main_id, i, process_func, *process_param)
    preprocess_process_pool.shutdown(wait=True)
    # logging.info(f"Merging...")
    result = list()
    for i in range(num_sub_worker):
        with open(f"{ProjectConfig.tmp_path}/{version}_{main_id}_{i}", "rb") as f:
            result += _pickle.load(f)
    return result


def step1(project, version, project_compiled_path, compiled_path):
    try:
        logging.info(f"Load project tokenizer in {project}.")
        tokenizer = Tokenizer.from_file(f"{ProjectConfig.pretrain_tokenizer_path}")
        logging.info(f"Load raw method call in {project}-{version}.")
        with open(f"{compiled_path}/{ProjectConfig.method_call_filename}", "rb") as f:
            raw_data: dict = _pickle.load(f)
        vectors = dict()

        logging.info(f"Process method call data in {project}-{version}.")

        lst = list(raw_data.values())
        values = batch_vector(lst, version, process_word2vec, model, model2, tokenizer,
                              sub_task_func=sub_task_method_call)
        vectors = dict(zip(raw_data.keys(), values))
        with open(f"{compiled_path}/{ProjectConfig.vector_method_call_filename}", "wb+") as f:
            _pickle.dump(vectors, f)

        logging.info(f"Load method attribute in {project}-{version}.")
        with open(f"{compiled_path}/{ProjectConfig.raw_method_attributes_filename}", "rb") as f:
            raw_data: dict = _pickle.load(f)

        logging.info(f"Process method attribute in {project}-{version}.")
        lst1 = list()
        lst2 = list()
        lst3 = list()
        for values in raw_data.values():
            lst1.append(values[0])
            lst2.append(values[1])
            lst3.append(values[2])
        values1 = batch_vector(lst1, version, process_word2vec, model, tokenizer)
        values2 = batch_vector(lst2, version, process_word2vec, model, tokenizer)
        values3 = batch_vector(lst3, version, process_node2vec, model2)
        del lst1, lst2, lst3
        vectors = dict()
        for i, key in enumerate(raw_data.keys()):
            vectors[key] = values1[i], values2[i], values3[i]
        del values1, values2, values3
        with open(f"{compiled_path}/{ProjectConfig.vector_method_attribute_filename}", "wb+") as f:
            _pickle.dump(vectors, f)

        logging.info(f"Load test attribute in {project}-{version}.")
        with open(f"{compiled_path}/{ProjectConfig.raw_test_attributes_filename}", "rb") as f:
            raw_data: dict = _pickle.load(f)

        logging.info(f"Process test attribute data in {project}-{version}.")
        lst1 = list()
        lst2 = list()
        for values in raw_data.values():
            lst1.append(values[0])
            lst2.append(values[1])
        values1 = batch_vector(lst1, version, process_word2vec, model, tokenizer)
        values2 = batch_vector(lst2, version, process_word2vec, model, tokenizer)
        del lst1, lst2
        vectors = dict()
        for i, key in enumerate(raw_data.keys()):
            vectors[key] = values1[i], values2[i]
        del values1, values2
        logging.info(f"Process method2[pass/fail]test_attribute data in {project}-{version}.")
        with open(f"{compiled_path}/{ProjectConfig.method2test_filename}", "rb") as f:
            method2test = _pickle.load(f)
        fail_set = set()
        with open(f"{compiled_path}/{ProjectConfig.failing_tests_filename}", "r") as f:
            while (line := f.readline()) != "":
                fail_set.add(line.strip())
        pass_dict = dict()
        fail_dict = dict()
        for method_key, target_test in method2test.items():
            target_test_set = set(target_test)
            for test_key, test_vector in vectors.items():
                if test_key in target_test_set:
                    if test_key in fail_set:
                        target = fail_dict.get(method_key)
                        if target is None:
                            fail_dict[method_key] = [test_vector]
                        else:
                            fail_dict[method_key] = target + [test_vector]
                    else:
                        target = pass_dict.get(method_key)
                        if target is None:
                            pass_dict[method_key] = [test_vector]
                        else:
                            pass_dict[method_key] = target + [test_vector]
        with open(f"{compiled_path}/{ProjectConfig.vector_pass_test_attribute_filename}", "wb+") as f:
            _pickle.dump(pass_dict, f)
        with open(f"{compiled_path}/{ProjectConfig.vector_fail_test_attribute_filename}", "wb+") as f:
            _pickle.dump(fail_dict, f)
        del fail_set, pass_dict, fail_dict

        logging.info(f"Load history in {project}-{version}.")
        with open(f"{compiled_path}/{ProjectConfig.methods_history_filename}", "rb") as f:
            raw_data: dict = _pickle.load(f)

        logging.info(f"Process history data in {project}-{version}.")
        lst = list(raw_data.values())
        values = batch_vector(lst, version, process_word2vec, model, model2, tokenizer,
                              sub_task_func=sub_task_method_call)
        vectors = dict(zip(raw_data.keys(), values))
        with open(f"{compiled_path}/{ProjectConfig.vector_method_history_filename}", "wb+") as f:
            _pickle.dump(vectors, f)
    except Exception:
        logging.error(traceback.format_exc())
        return False
    return True


def run(project, version):
    work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    project_compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/"
    if not step1(project, version, project_compiled_path, compiled_path):
        logging.error(f"Error at {project}-{version} when running step1.")
    logging.info(f"Finished {project}-{version}.")


if __name__ == '__main__':
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)
