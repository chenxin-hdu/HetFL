import csv
import os
import logging
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
    for _, root in java_tree:
        if isinstance(root, javalang.tree.Declaration):
            for _, node in root:
                if isinstance(node, MethodDeclaration) or isinstance(node, ConstructorDeclaration):
                    method_signature = Utils.get_method_signature(node)
                    method_path = f"{current_path}${node.name}"
                    result.append(f"{method_path} {method_signature}")
    return result


def step1(project, version, work_path, compiled_path):
    trigger_tests = list()
    with open(f"{compiled_path}/{ProjectConfig.raw_tests_trigger_filename}", "r") as f:
        for line in f.readlines():
            trigger_tests.append(line.strip().replace("::", "$"))
    logging.info(f"Get all trigger method in {project}-{version}.")
    relevant_class_paths = list()
    for trigger_test in trigger_tests:
        _, shortname = trigger_test.split("$")
        with open(f"{compiled_path}/{ProjectConfig.coverage_test2method_dirname}/{shortname}.txt", "r") as f:
            for line in f.readlines():
                relevant_class_paths.append(line.strip())
    write_file = open(f"{compiled_path}/{ProjectConfig.relevant_method_list}", "w+")
    for relevant_class_path in relevant_class_paths:
        write_file.write(relevant_class_path + os.linesep)
    write_file.close()
    return True

def step2(project, version, work_path, compiled_path):
    with open(f"{compiled_path}/{ProjectConfig.source_root_dir_file}", "r") as f:
        source_root = f.readline().strip()
    logging.info(f"Get all method in {project}-{version}.")
    source_path = f"{work_path}/{source_root}"
    write_file = open(f"{compiled_path}/{ProjectConfig.all_method_list}", "w+")
    for root, dirs, files in os.walk(source_path):
        current_path = os.path.relpath(root, source_path).replace("/", ".")
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                results = get_methods_from_file(file_path, current_path + f".{file[:-5]}")
                write_file.write("".join(results))
    write_file.close()
    return True



def run(project, version):
    work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    if not step1(project, version, work_path, compiled_path):
        logging.error(f"Error at {project}-{version} when running step1.")
    if not step2(project, version, work_path, compiled_path):
        logging.error(f"Error at {project}-{version} when running step2.")


if __name__ == '__main__':
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)
