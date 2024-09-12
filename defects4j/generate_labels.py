import csv
import os
import logging
import pickle
import sys
import xml.etree.ElementTree as ET
import javalang
from javalang import parse
from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
except ImportError:
    from ..config import ProjectConfig
    from .utils import Utils


def try_replace_by_dict(value, _dict: dict):
    if r := _dict.get(value):
        return r
    else:
        return value


def get_methods_from_file(file_path, current_path):
    result = []
    java_tree: CompilationUnit
    with open(file_path, "r", encoding='ISO-8859-1') as f:
        file_content = f.read()
        java_tree = javalang.parse.parse(file_content)
    dict_import = dict()
    for _, root in java_tree:
        if isinstance(root, javalang.tree.Import):
            path: str = root.path
            import_obj_name = path[path.rindex(".") + 1:]
            dict_import[import_obj_name] = path
        if isinstance(root, javalang.tree.Declaration):
            for _, node in root:
                if isinstance(node, MethodDeclaration) or isinstance(node, ConstructorDeclaration):
                    method_signature = Utils.get_method_signature(node, dict_import)
                    method_path = f"{current_path}${node.name}"
                    result.append(f"{method_path} {method_signature}{os.linesep}")
    return result


def step1(project, version, work_path, compiled_path):
    logging.info(f"Load relevent method list in {project}-{version}.")
    labels = dict()
    f = open(f"{compiled_path}/{ProjectConfig.relevant_method_list}", "r")
    while (line := f.readline()) != "":
        labels[line.strip()] = False
    f.close()
    f = open(f"{compiled_path}/{ProjectConfig.bug_method_filename}", "r")
    flag = True
    while (line := f.readline()) != "":
        if labels.get(line.strip()) is not None:
            labels[line.strip()] = True
            flag = False
        else:
            labels[line.strip()] = True
            flag = False
    if flag:
        logging.warning(f"No bug method found in relevant method list in {project}-{version}.")
    f.close()
    f = open(f"{compiled_path}/{ProjectConfig.labels_filename}", "wb+")
    pickle.dump(labels, f)
    f.close()
    f = open(f"{compiled_path}/{ProjectConfig.relevant_method_list}", "w+")
    for t in labels.keys():
        f.write(t+os.linesep)
    f.close()
    return True


def run(project, version):
    work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    if not step1(project, version, work_path, compiled_path):
        logging.error(f"Error at {project}-{version} when running step1.")


if __name__ == '__main__':
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)
