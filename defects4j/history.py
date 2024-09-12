import os
import logging
import pickle
import re
import sys
import traceback

import javalang
import subprocess
import networkx as nx
from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
except ImportError:
    from ..config import ProjectConfig
    from .utils import Utils


def system_retry(command, times=3):
    current_times = 0
    while current_times <= times:
        if os.system(command) != 0:
            current_times += 1
            logging.warning(f"Retry={current_times} {command}.")
        else:
            return True
    return False


token_stat = (javalang.tree.VariableDeclaration, javalang.tree.LocalVariableDeclaration,
              javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration,
              javalang.tree.IfStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement,
              javalang.tree.ForStatement, javalang.tree.AssertStatement, javalang.tree.SwitchStatement,
              javalang.tree.BreakStatement, javalang.tree.ContinueStatement, javalang.tree.ReturnStatement,
              javalang.tree.ThrowStatement, javalang.tree.TryStatement, javalang.tree.StatementExpression,
              javalang.tree.CatchClause, javalang.tree.SwitchStatementCase)


class IndexProvider:
    def __init__(self):
        self.idx = 0

    def get(self):
        self.idx += 1
        return self.idx


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


def get_node_ast(node, node_str=None, index_provider=None, graph=None):
    if index_provider is None:
        index_provider = IndexProvider()
    if graph is None:
        graph = nx.Graph()
        graph.add_node("root")
        node_str = "root"
    if hasattr(node, "body") and node.body is not None:
        for body in node.body:
            body_type = type(body)
            if body_type in token_stat:
                body_str = str(body_type)
                body_str = f"{body_str[body_str.rindex('.') + 1:-2]} {index_provider.get()}"
                graph.add_node(body_str)
                graph.add_edge(node_str, body_str)
                if hasattr(body, "body") and body.body is not None:
                    get_node_ast(body, node_str=body_str, index_provider=index_provider, graph=graph)
    return graph


def get_methods_from_file(content, current_path, java_tree: CompilationUnit):
    result = dict()
    for _, root in java_tree:
        if isinstance(root, javalang.tree.Declaration):
            for _, node in root:
                if isinstance(node, MethodDeclaration) or isinstance(node, ConstructorDeclaration):
                    method_signature = Utils.get_method_signature(node)
                    method_path = f"{current_path}${node.name}"
                    method_key = f"{method_path} {method_signature}"
                    method_code = get_code_block(content, node)
                    method_name = node.name
                    # noinspection PyTypeChecker
                    method_ast_graph = get_node_ast(node)
                    result[method_key] = (method_name, method_code, method_ast_graph)
    return result


def step1(project, version, work_path, compiled_path):
    # create virtual env
    os.system(f"rm -rf {work_path}")
    Utils.create_path(work_path)
    if not system_retry(f"defects4j checkout -p {project} -v {version}b -w {work_path}"):
        logging.error(f"Checkout for virtual env failed at {project}-{version}.")
        return False
    return True


def step2(project, version, virtual_work_path, compiled_path):
    os.chdir(virtual_work_path)
    logging.info(f"Load all method info in {project}-{version}.")
    with open(f"{compiled_path}/{ProjectConfig.source_root_dir_file}", "r") as f:
        source_root = f.readline().strip()
    java_files = list()
    source_path = f"{virtual_work_path}/{source_root}"
    historys = {}
    logging.info(f"Load head branch methods in {project}-{version}.")
    relevant_class_files = list()
    relevant_class_paths = list()
    with open(f"{compiled_path}/{ProjectConfig.classes_relevant_filename}", "r") as f:
        for line in f.readlines():
            relevant_class_files.append(f"{source_root}/{line.strip().replace('.', '/')}.java")
            relevant_class_paths.append(line.strip())
    logging.info(f"Get head branch methods in {project}-{version}.")
    for file_path, current_path in zip(relevant_class_files, relevant_class_paths):
        if file_path.endswith(".java"):
            logging.info(f"Get head branch methods for {file_path} in {project}-{version}.")
            if not os.path.exists(file_path):
                continue
            java_files.append(file_path)
            with open(file_path, "r", encoding='ISO-8859-1') as f:
                file_content = f.read()
                java_tree = javalang.parse.parse(file_content)
            _histories = get_methods_from_file(file_content, current_path, java_tree)
            for key, value in _histories.items():
                target = historys.get(key)
                if target is None:
                    historys[key] = [value]
                else:
                    historys[key].append(value)
    for file_path in java_files:
        logging.info(f"Get history for {file_path} in {project}-{version}.")
        result = subprocess.run([f'git log --pretty=format:"%H" {file_path} '], capture_output=True, text=True,
                                shell=True)
        if result.stdout != "":
            commits = result.stdout.splitlines()
            del result
            for i in range(1, len(commits)):
                os.system(f"rm -rf {file_path}")
                if os.system(f"git checkout {commits[i]} {file_path}") != 0 or not os.path.exists(file_path):
                    continue
                with open(file_path, "r",encoding='ISO-8859-1') as f:
                    try:
                        file_content = f.read()
                        java_tree = javalang.parse.parse(file_content)
                    except:
                        continue
                current_path = os.path.relpath(file_path, source_root)[:-5].replace("/", ".")
                _histories = get_methods_from_file(file_content, current_path, java_tree)
                for key, value in _histories.items():
                    target = historys.get(key)
                    if target is None:
                        historys[key] = [value]
                    elif value[1] != target[-1][1]:
                        historys[key].append(value)
    for key in list(historys.keys()):
        if (value := historys.get(key)) is not None:
            if len(value) <= 1:
                historys.pop(key)
            else:
                historys[key] = value[1:]
    with open(f"{compiled_path}/{ProjectConfig.methods_history_filename}", "wb+") as f:
        pickle.dump(historys, f)
    return True


def run(project, version):
    try:
        work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
        compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
        virtual_work_path = f"{ProjectConfig.path_dataset_home}/projects/" \
                            f"{project}/{version}/{ProjectConfig.project_history_temp_dirname}"
        if not step1(project, version, virtual_work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step1.")
            exit()
        if not step2(project, version, virtual_work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step2.")
        logging.info(f"Finished: {project}-{version}.")
    except:
        logging.error(traceback.format_exc())

def main():
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)

if __name__ == '__main__':
    main()

