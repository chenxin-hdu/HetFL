import sys

from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration
import javalang
import logging
import os
import pickle
import re
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
except ImportError:
    from ..config import ProjectConfig
    from .utils import Utils


def delete_comments(content):
    out = re.sub(r'/\*.*?\*/', '', content, flags=re.S)
    out = re.sub(r'(//.*)', '', out)
    out = out.replace('@Override', '').strip()
    return out


def get_node_body_text_by_node_start(content: str, start: int):
    lines = content.splitlines(True)
    string = "".join(lines[start - 1:])
    left = string.count("{")
    right = string.count("}")
    if right - left == 1:
        p = string.rfind("}")
        string = string[:p]
    return string


def get_code_block(content, node):
    return delete_comments(get_node_body_text_by_node_start(content, int(node.position.line)))


def get_tests_attr(content, current_path, java_tree: CompilationUnit):
    result = dict()
    for _, root in java_tree:
        if isinstance(root, javalang.tree.Declaration):
            for _, node in root:
                if isinstance(node, MethodDeclaration):
                    test_path = f"{current_path}${node.name}"
                    test_key = f"{test_path}"
                    test_code = get_code_block(content, node)
                    test_name = node.name
                    result[test_key] = (test_name, test_code)
    return result


def step1(project, version, work_path, compiled_path):
    try:
        with open(f"{compiled_path}/{ProjectConfig.test_root_dir_file}", "r") as f:
            test_root = f.readline().strip()
        logging.info(f"Get all test attributes in {project}-{version}.")
        source_path = f"{work_path}/{test_root}"
        results = dict()
        for root, dirs, files in os.walk(source_path):
            current_path = os.path.relpath(root, source_path).replace("/", ".")
            for file in files:
                if file.endswith(".java"):
                    logging.info(f"Analysis {file} in {project}-{version}.")
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding='ISO-8859-1') as f:
                        file_content = f.read()
                        try:
                            java_tree = javalang.parse.parse(file_content)
                        except:
                            continue
                    results.update(get_tests_attr(file_content, current_path + f".{file[:-5]}", java_tree))
        with open(f"{compiled_path}/{ProjectConfig.raw_test_attributes_filename}", "wb+") as f:
            pickle.dump(results, f)
    except Exception:
        logging.error(traceback.format_exc())
        return False
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
