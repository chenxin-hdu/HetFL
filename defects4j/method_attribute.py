import sys

from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration
import javalang
import logging
import networkx as nx
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


def get_methods_attr(content, current_path, java_tree: CompilationUnit, trigger_set: set = None):
    result = dict()
    for _, root in java_tree:
        if isinstance(root, javalang.tree.Declaration):
            for _, node in root:
                if isinstance(node, MethodDeclaration) or isinstance(node, ConstructorDeclaration):
                    method_signature = Utils.get_method_signature(node)
                    method_path = f"{current_path}${node.name}"
                    method_key = f"{method_path} {method_signature}"
                    if trigger_set is not None:
                        if method_key not in trigger_set:
                            continue
                    method_code = get_code_block(content, node)
                    method_name = node.name
                    # noinspection PyTypeChecker
                    method_ast_graph = get_node_ast(node)
                    result[method_key] = (method_name, method_code, method_ast_graph)
    return result


def step1(project, version, work_path, compiled_path):
    try:
        os.chdir(work_path)
        with open(f"{compiled_path}/{ProjectConfig.source_root_dir_file}", "r") as f:
            source_root = f.readline().strip()
        logging.info(f"Get all method attributes in {project}-{version}.")
        results = dict()
        relevant_class_files = list()
        relevant_class_paths = list()
        trigger_keys = set()
        with open(f"{compiled_path}/{ProjectConfig.relevant_method_list}", "r") as f:
            while (line := f.readline()) != "":
                trigger_keys.add(line.strip())
                t = line.strip().split("$")[0]
                relevant_class_files.append(f"{source_root}/{t.replace('.', '/')}.java")
                relevant_class_paths.append(t)
        bfs_path = set()
        for file_path, current_path in zip(relevant_class_files, relevant_class_paths):
            if current_path in bfs_path:
                continue
            bfs_path.add(current_path)
            logging.info(f"Get method attributes for {current_path} in {project}-{version}.")
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding='ISO-8859-1') as f:
                file_content = f.read()
                java_tree = javalang.parse.parse(file_content)
            result = get_methods_attr(file_content, current_path, java_tree, trigger_keys)
            results.update(result)
        with open(f"{compiled_path}/{ProjectConfig.raw_method_attributes_filename}", "wb+") as f:
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