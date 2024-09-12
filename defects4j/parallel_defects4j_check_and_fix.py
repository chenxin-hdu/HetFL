import traceback

from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration
import concurrent.futures
import javalang
import logging
import os
import sys
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
except ImportError:
    from ..config import ProjectConfig
from defects4j.utils import Utils

process_pool_size = 128

dict_type_signature_symbol2full_name = {
    "B": "byte",
    "C": "char",
    "D": "double",
    "F": "float",
    "I": "int",
    "J": "long",
    "O": "object",
    "S": "short",
    "V": "void",
    "Z": "boolean",
}


def decode_origin_type_signature_symbol(signature):
    if len(signature) == 1:
        return dict_type_signature_symbol2full_name.get(signature)
    decode_signature = []
    array_flag = False
    for key in signature:
        if key == "[":
            array_flag = True
            continue
        decode_key = dict_type_signature_symbol2full_name.get(key)
        if array_flag:
            decode_signature.append("[" + decode_key)
        else:
            decode_signature.append(decode_key)
        array_flag = False

    decode_signature = ";".join(decode_signature)
    return decode_signature


def decode_type_signature_symbol(signature):
    decode_param = ""
    if signature.startswith("["):
        decode_param += "["
        signature = signature[1:]
    if signature.startswith("L"):
        signature = signature[1:]
        if signature == "java/lang/String":
            signature = "String"
        else:
            signature = signature.replace("/", ".")
        decode_param += signature.strip(";")
    else:
        key = signature[0]
        if key in dict_type_signature_symbol2full_name:
            decode_param += decode_origin_type_signature_symbol(key)
        if len(signature) > 1:
            decode_param += f";{decode_type_signature_symbol(signature[1:])}"
    return decode_param


def decode_coverage_signature(coverage_signature):
    parts = coverage_signature.split(")")
    params = parts[0][1:].split(";")
    return_string = parts[1]
    decode_params = []
    for param in params:
        if param == "":
            continue
        decode_params.append(decode_type_signature_symbol(param))
    decode_params = ";".join(decode_params)
    return_string = decode_type_signature_symbol(return_string)
    decode_signature = f"({decode_params}){return_string}"
    return decode_signature


def system_retry(command, times=3):
    current_times = 0
    while current_times <= times:
        if os.system(command) != 0:
            current_times += 1
            logging.warning(f"Retry={current_times} {command}.")
        else:
            return True
    return False


def exports(key, save_filepath, work_path):
    system_retry(f"defects4j export -p {key} -o {save_filepath} -w {work_path}")


def exist_and_contain_content(filepath):
    return os.path.exists(filepath) and os.stat(filepath).st_size > 0


def exports_if_not_exists(key, save_filepath, work_path):
    if not exist_and_contain_content(save_filepath):
        exports(key, save_filepath, work_path)


def step1(project, version, work_path, compiled_path):
    if os.path.exists(work_path):
        os.system(f"rm -rf {work_path}")
    logging.info(f"Checkout {project}-{version}.")
    if not system_retry(f"defects4j checkout -p {project} -v {version}b -w {work_path}"):
        logging.error(f"Checkout failed at {project}-{version}.")
        return False

    exports_if_not_exists("classes.relevant", f"{compiled_path}/{ProjectConfig.classes_relevant_filename}", work_path)

    exports_if_not_exists("tests.trigger", f"{compiled_path}/{ProjectConfig.raw_tests_trigger_filename}", work_path)

    exports_if_not_exists("classes.modified", f"{compiled_path}/{ProjectConfig.classes_modified_filename}", work_path)

    exports_if_not_exists("dir.src.classes", f"{compiled_path}/{ProjectConfig.source_root_dir_file}", work_path)

    exports_if_not_exists("dir.src.tests", f"{compiled_path}/{ProjectConfig.test_root_dir_file}", work_path)
    return True


def step2(project, version, work_path, compiled_path):
    all_tests_filepath = f"{compiled_path}/ALL_TESTS"
    failing_tests_filepath = f"{compiled_path}/FAILING_TESTS"
    logging.info(f"Compile {project}-{version}.")
    if not system_retry(f"defects4j compile -w {work_path}"):
        logging.error(f"Compile failed at {project}-{version}.")
        return False
    logging.info(f"Test {project}-{version}.")
    if not system_retry(f"defects4j test -r -w {work_path}"):
        logging.error(f"Test failed at {project}-{version}.")
        return False
    os.system(f"cp -f {work_path}/all_tests {compiled_path}/ALL_TESTS")
    os.system(f"cp -f {work_path}/failing_tests {compiled_path}/FAILING_TESTS")
    return True


def step3(project, version, work_path, compiled_path):
    if exist_and_contain_content(f"{compiled_path}/{ProjectConfig.failing_tests_filename}") and \
            exist_and_contain_content(
                f"{compiled_path}/{ProjectConfig.pass_tests_filename}") and exist_and_contain_content(
        f"{compiled_path}/{ProjectConfig.all_tests_filename}") and exist_and_contain_content(
        f"{compiled_path}/{ProjectConfig.tests_trigger_filename}"):
        logging.info(f"Tests info has already been processed in {project}-{version}.")
        return True
    logging.info(f"Processe tests info in {project}-{version}.")
    failing_tests = set()
    with open(f"{compiled_path}/FAILING_TESTS", "r") as f:
        while (line := f.readline()) != '':
            if line.startswith("---"):
                line = line[4:].strip()
                full_path = line.replace("::", "$")
                failing_tests.add(full_path)
    all_tests = set()
    with open(f"{compiled_path}/ALL_TESTS", "r") as f:
        while (line := f.readline()) != '':
            line = line.replace(" ", "_")
            method, path = line.split("(")
            path = path[:-2]
            full_path = f"{path}${method}"
            all_tests.add(full_path)
    pass_tests = all_tests.difference(failing_tests)
    with open(f"{compiled_path}/{ProjectConfig.failing_tests_filename}", "w+") as f:
        for line in failing_tests:
            f.writelines(line + os.linesep)
    with open(f"{compiled_path}/{ProjectConfig.pass_tests_filename}", "w+") as f:
        for line in pass_tests:
            f.writelines(line + os.linesep)
    with open(f"{compiled_path}/{ProjectConfig.all_tests_filename}", "w+") as f:
        for line in all_tests:
            f.writelines(line + os.linesep)
    trigger_tests = []
    with open(f"{compiled_path}/{ProjectConfig.raw_tests_trigger_filename}", "r") as f:
        while (line := f.readline()) != '':
            line = line.strip().replace("::", "$")
            trigger_tests.append(line)
    with open(f"{compiled_path}/{ProjectConfig.tests_trigger_filename}", "w+") as f:
        for trigger_test in trigger_tests:
            f.write(trigger_test + os.linesep)
    return True


def step4(project, version, work_path, compiled_path):
    os.chdir(work_path)
    logging.info(f"Run git log in {project}-{version}.")
    if os.system(f'git log --pretty=format:"%H %s"> {compiled_path}/{ProjectConfig.git_log_filename}') != 0:
        logging.error(f"Run git log failed in {project}-{version}.")
        return False
    logging.info(f"Get commit id of buggy version and fixed version in {project}-{version}.")
    commit_id_buggy = None
    commit_id_fixed = None
    with open(f"{compiled_path}/{ProjectConfig.git_log_filename}", "r") as f:
        for line in f.readlines():
            commit_id, des = line.strip().split(" ")
            if des.endswith("FIXED_VERSION"):
                commit_id_fixed = commit_id
            elif des.endswith("BUGGY_VERSION"):
                commit_id_buggy = commit_id
            if commit_id_buggy and commit_id_fixed:
                break
    logging.info(f"Run git diff in {project}-{version} with {commit_id_buggy} and {commit_id_fixed}.")
    diff_save_path = f"{compiled_path}/{ProjectConfig.git_diff_buggy_fix_filename}"
    if os.system(f"git diff -U0 {commit_id_buggy} {commit_id_fixed} > {diff_save_path}") != 0:
        logging.error(f"Run git diff failed in {project}-{version}.")
        return False
    logging.info(f"Load modified filename in {project}-{version}.")
    modified_files = set()
    with open(f"{compiled_path}/{ProjectConfig.classes_modified_filename}", "r") as f:
        while (line := f.readline()) != '':
            modified_files.add(f'{line.strip()}')
    bug_path = None
    bug_line = None
    logging.info(f"Get modified line from {diff_save_path} in {project}-{version}.")
    with open(f"{compiled_path}/{ProjectConfig.source_root_dir_file}", "r") as f:
        source_root = f.readline().strip()
    bugs = []
    with open(diff_save_path, "r", ) as f:
        diff_filepath = None
        while True:
            try:
                if (line := f.readline()) == '':
                    break
            except Exception:
                logging.warning(f"In {project}-{version}: Failed to read certain line of {diff_save_path}.{os.linesep}"
                                f"{traceback.format_exc()}")
                break
            if line.startswith("diff"):
                parts = line.strip().split(" ")
                origin_path = parts[-1]
                try:
                    if not origin_path.endswith(".java"):
                        continue
                    start_index = origin_path.index("/", 1)
                except ValueError:
                    diff_filepath = None
                else:
                    diff_filepath = os.path.relpath(origin_path[start_index + 1:-5], source_root).replace("/", ".")
            elif line.startswith("--- /dev/null"):
                diff_filepath = None
            elif line.startswith("@@") and diff_filepath is not None:
                parts = line.strip().split(" ")
                change_line = parts[1][1:]
                try:
                    change_line = change_line[:change_line.index(",")]
                except ValueError:
                    pass
                diff_filepath = os.path.relpath(origin_path[start_index + 1:-5], source_root).replace("/", ".")
                if diff_filepath in modified_files:
                    bug_path = diff_filepath
                    bug_line = int(change_line)
                    bugs.append((bug_path, bug_line))
    logging.info(f"Get bug method by bug file and line in {project}-{version}.")
    fw = open(f"{compiled_path}/{ProjectConfig.bug_method_filename}", "w+")
    found_bugs = set()
    for bug_path, bug_line in bugs:
        bug_filepath = f"{work_path}/{source_root}/{bug_path.replace('.', '/')}.java"
        java_tree: CompilationUnit
        with open(bug_filepath, "r", encoding='ISO-8859-1') as f:
            file_content = f.read()
            java_tree = javalang.parse.parse(file_content)
        closest_method_line = -1
        method = None
        method_name = "ErrorErrorError"
        for _, root in java_tree:
            if isinstance(root, javalang.tree.Declaration):
                for _, node in root:
                    if isinstance(node, MethodDeclaration) or isinstance(node, ConstructorDeclaration):
                        current_method_line = node.position[0]
                        if bug_line >= current_method_line > closest_method_line:
                            closest_method_line = current_method_line
                            method = node
                            if hasattr(method, "name"):
                                if method.name is None:
                                    logging.warning(f"Error method node: {method}")
                                else:
                                    method_name = method.name
                            else:
                                logging.warning(f"Error method node: {method}")
                                method_name = "ErrorErrorError"
        if not hasattr(method, "name"):
            continue
        if method.name == "ErrorErrorError":
            continue
        method_path = f"{bug_path}${method_name}"
        method_signature = Utils.get_method_signature(method)
        if f"{method_path} {method_signature}" not in found_bugs:
            found_bugs.add(f"{method_path} {method_signature}")
            logging.info(f"Find bug method {method_path} {method_signature} in {project}-{version}.")
            fw.write(f"{method_path} {method_signature}{os.linesep}")
    fw.close()
    return True


def step5(project, version, work_path, compiled_path):
    Utils.create_path(f"{compiled_path}/{ProjectConfig.coverage_dirname}")
    Utils.create_path(f"{compiled_path}/{ProjectConfig.coverage_failing_dirname}")
    instrument_classes = f"{compiled_path}/{ProjectConfig.classes_relevant_filename}"
    with open(f"{compiled_path}/{ProjectConfig.failing_tests_filename}", 'r') as f:
        for full_path in f.readlines():
            try:
                path, method = full_path.strip().split("$")
            except:
                continue
            xml_file = f"{compiled_path}/{ProjectConfig.coverage_dirname}/{method}.xml"
            skip_coverage = True
            if os.path.exists(xml_file):
                logging.info(f"Check {method} in {project}-{version}.")
                if not check_coverage(xml_file):
                    skip_coverage = False
                    logging.info(f"Invalid {xml_file} in {project}-{version}.")
            else:
                skip_coverage = False
            method_failing_filepath = f"{compiled_path}/{ProjectConfig.coverage_failing_dirname}/{method}.txt"
            if not exist_and_contain_content(method_failing_filepath):
                skip_coverage = False
            if not skip_coverage:
                logging.info(f"Coverage for {method} in {project}-{version}.")
                failing_tests_filepath = f"{work_path}/failing_tests"
                if os.path.exists(failing_tests_filepath):
                    os.system(f"rm -f {failing_tests_filepath}")
                if not system_retry(f"defects4j coverage -w {work_path} -t {path}::{method} -i {instrument_classes}"):
                    logging.error(f"Coverage failed in {project}-{version} at {full_path}. Continue.")
                    continue
                os.system(
                    f"cp -f {work_path}/coverage.xml {compiled_path}/{ProjectConfig.coverage_dirname}/{method}.xml")
                os.system(
                    f"cp -f {failing_tests_filepath} {method_failing_filepath}")
    with open(f"{compiled_path}/{ProjectConfig.pass_tests_filename}", 'r') as f:
        for full_path in f.readlines():
            try:
                path, method = full_path.strip().split("$")
            except:
                continue
            xml_file = f"{compiled_path}/{ProjectConfig.coverage_dirname}/{method}.xml"
            skip_coverage = True
            if os.path.exists(xml_file):
                logging.info(f"Check {method} in {project}-{version}.")
                if not check_coverage(xml_file):
                    skip_coverage = False
                    logging.info(f"Invalid {xml_file} in {project}-{version}.")
            else:
                skip_coverage = False
            if not skip_coverage:
                logging.info(f"Coverage for {method} in {project}-{version}.")
                failing_tests_filepath = f"{work_path}/failing_tests"
                if os.path.exists(failing_tests_filepath):
                    os.system(f"rm -f {failing_tests_filepath}")
                if not system_retry(f"defects4j coverage -w {work_path} -t {path}::{method} -i {instrument_classes}"):
                    logging.error(f"Coverage failed in {project}-{version} at {full_path}. Continue.")
                    continue
                os.system(
                    f"cp -f {work_path}/coverage.xml {compiled_path}/{ProjectConfig.coverage_dirname}/{method}.xml")
    return True


def check_coverage(xml_filepath):
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
    except Exception:
        return False
    return True


def run(project, version):
    work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    Utils.create_path(work_path)
    Utils.create_path(compiled_path)
    try:
        if os.path.exists(f'{compiled_path}/checked'):
            # os.system(f"rm -f {compiled_path}/checked")
            logging.info(f"{project}-{version} has been checked.")
            # return
        if not step1(project, version, work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step1.")
        if not step2(project, version, work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step2.")
        if not step3(project, version, work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step3.")
        if not step4(project, version, work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step4.")
        if not step5(project, version, work_path, compiled_path):
            logging.error(f"Error at {project}-{version} when running step5.")
        open(f'{compiled_path}/checked', 'w+').close()
        logging.info(f"Finished: {project}-{version}.")
    except Exception:
        logging.error(f"In {project}-{version}: " + traceback.format_exc())


def main():
    process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=process_pool_size)
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            process_pool.submit(run, __project, __version)


if __name__ == '__main__':
    main()
