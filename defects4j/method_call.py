import sys

from javalang.tree import CompilationUnit, MethodDeclaration, ConstructorDeclaration
import logging
import os
import pickle
import traceback
import networkx as nx # need it
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


def exports(key, save_filepath, work_path):
    system_retry(f"defects4j export -p {key} -o {save_filepath} -w {work_path}")


def exist_and_contain_content(filepath):
    return os.path.exists(filepath) and os.stat(filepath).st_size > 0


def init_java_callgraph2(work_path):
    Utils.create_path(f"{work_path}/{ProjectConfig.java_callgraph2_dirname}")
    os.system(
        f"cp -r {ProjectConfig.java_callgraph2_origin_dir_path}/* {work_path}/{ProjectConfig.java_callgraph2_dirname}")


def init_virtual_env(project, work_path, virtual_work_path, compiled_path):
    os.chdir(work_path)
    if project in ["Closure", "Chart"]:
        class_path = "build"
        if os.path.exists(f"{work_path}/build/classes"):
            class_path = "build/classes"
    elif project in ["Compress", "JxPath","JacksonDatabind"]:
        class_path = "target/classes"
    else:
        class_path_filepath = f"{compiled_path}/{ProjectConfig.classes_path_filename}"
        # if not exist_and_contain_content(class_path_filepath):
        #     if os.path.exists(work_path):
        #         os.system(f"rm -rf {work_path}")
        #     exports("cp.compile", class_path_filepath, work_path)
        with open(class_path_filepath, "r") as f:
            class_path = f.readline().strip()
    system_retry(f"jar -cf {virtual_work_path}/target.jar -C {class_path} .")


def config_java_callgraph2(work_path, virtual_work_path):
    config_path = f"{work_path}/{ProjectConfig.java_callgraph2_dirname}/_javacg_config/jar_dir.properties"
    with open(config_path, "w+") as f:
        f.write(f"{virtual_work_path}/target.jar" + os.linesep)


def run_java_callgraph2(work_path, virtual_work_path):
    os.system(f"sh {work_path}/{ProjectConfig.java_callgraph2_dirname}/run.sh")


def decode_param_type(param: str):
    result = ""
    while param.endswith("[]"):
        result += "["
        param = param[:-2]
    param = param.split(".")[-1]
    result += param
    return result


def analysis(work_path, virtual_work_path, compiled_path):
    method_call_info = dict()
    method_be_called_info = dict()
    fr = open(f"{virtual_work_path}/target.jar-output_javacg/method_call.txt", "r")
    while (line := fr.readline()) != "":
        parts = line.strip().split("\t")
        current_method_parts = parts[1].split(":")
        call_method_parts = parts[2].split(":")
        current_method_name = current_method_parts[0]
        current_method_params = current_method_parts[1].split("(")[1][:-1]
        call_method_name = call_method_parts[0].split(")")[-1]
        call_method_params = call_method_parts[1].split("(")[1][:-1]
        current_method_return = parts[4]
        call_method_return = parts[6]
        if not "$" in current_method_name:
            name = current_method_name.split(".")[-1]
            current_method_name += "$" + name
        if not "$" in call_method_name:
            name = call_method_name.split(".")[-1]
            call_method_name += "$" + name
        current_method_params = ";".join(
            [decode_param_type(param) for param in current_method_params.split(",")])
        call_method_params = ";".join(
            [decode_param_type(param) for param in call_method_params.split(",")])
        current_method_return = decode_param_type(current_method_return)
        call_method_return = decode_param_type(call_method_return)
        current_method_signature = f"{current_method_name} ({current_method_params}){current_method_return}"
        call_method_signature = f"{call_method_name} ({call_method_params}){call_method_return}"
        method_call_info[current_method_signature] = call_method_signature
        method_be_called_info[call_method_signature] = current_method_signature
    fr.close()
    return method_call_info, method_be_called_info


def _build(method_call_info: dict, method_be_called_info: dict, current_method, call_info: list):
    method = method_call_info.get(current_method)
    if method and method not in call_info:
        call_info.append(method)
        call_info = _build(method_call_info, method_be_called_info, method, call_info)
    method = method_be_called_info.get(current_method)
    if method and method not in call_info:
        call_info.append(method)
        call_info = _build(method_call_info, method_be_called_info, method, call_info)
    return call_info


def build(method_call_info: dict, method_be_called_info: dict, compiled_path):
    relevant_method_list = list()
    call_info = dict()
    with open(f"{compiled_path}/{ProjectConfig.relevant_method_list}", "r") as f:
        while (line := f.readline()) != '':
            relevant_method_list.append(line.strip())
    for method in relevant_method_list:
        _call_info = [method]
        _method = method_call_info.get(method)
        if _method:
            _call_info.append(_method)
            _call_info = _build(method_call_info, method_be_called_info, _method, _call_info)
        _method = method_be_called_info.get(method)
        if _method and _method not in _call_info:
            _call_info.append(_method)
            _call_info = _build(method_call_info, method_be_called_info, _method, _call_info)
        if len(_call_info) > 1:
            call_info[method] = _call_info[1:]
    return call_info


def step1(project, version, work_path, virtual_work_path, compiled_path):
    try:
        init_java_callgraph2(work_path)
        Utils.create_path(virtual_work_path)
        init_virtual_env(project, work_path, virtual_work_path, compiled_path)
        config_java_callgraph2(work_path, virtual_work_path)
        run_java_callgraph2(work_path, virtual_work_path)
        method_call_info, method_be_called_info = analysis(work_path, virtual_work_path, compiled_path)
        raw_call_info = build(method_call_info, method_be_called_info, compiled_path)
        with open(f"{compiled_path}/{ProjectConfig.raw_method_attributes_filename}", "rb") as f:
            method_attributes = pickle.load(f)
        call_info = dict()
        for key, value in raw_call_info.items():
            lst = list()
            for v in value:
                if (t := method_attributes.get(v)) != None:
                    lst.append(t)
                else:
                    # logging.warning(f"No key of {v}")
                    pass
            if len(lst) > 0:
                call_info[key] = lst
        with open(f"{compiled_path}/{ProjectConfig.method_call_filename}", "wb+") as f:
            pickle.dump(call_info, f)
    except Exception:
        logging.error(traceback.format_exc())
        return False
    return True


def run(project, version):
    work_path = f"{ProjectConfig.path_dataset_home}/projects/{project}/{version}"
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    virtual_work_path = f"{ProjectConfig.path_dataset_home}/projects/" \
                        f"{project}/{version}/{ProjectConfig.method_call_temp_dirname}"
    if not step1(project, version, work_path, virtual_work_path, compiled_path):
        logging.error(f"Error at {project}-{version} when running step1.")


def main():
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)


if __name__ == '__main__':
    main()
