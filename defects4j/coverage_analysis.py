import os
import logging
import sys
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
except ImportError:
    from ..config import ProjectConfig

from defects4j.utils import Utils

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


def decode_type_signature_symbol(signature, clz_path):
    decode_param = ""
    if signature.startswith("["):
        decode_param += "["
        signature = signature[1:]
    if signature.startswith("L"):
        signature = signature[1:]
        signature = signature.split("/")[-1]
        decode_param += signature.strip(";")
    else:
        key = signature[0]
        if key in dict_type_signature_symbol2full_name:
            decode_param += decode_origin_type_signature_symbol(key)
        if len(signature) > 1:
            decode_param += f";{decode_type_signature_symbol(signature[1:], clz_path)}"
    return decode_param


def decode_coverage_signature(coverage_signature, clz_path):
    parts = coverage_signature.split(")")
    params = parts[0][1:].split(";")
    return_string = parts[1]
    decode_params = []
    for param in params:
        if param == "":
            continue
        decode_params.append(decode_type_signature_symbol(param, clz_path))
    decode_params = ";".join(decode_params)
    return_string = decode_type_signature_symbol(return_string, clz_path)
    decode_signature = f"({decode_params}){return_string}"
    return decode_signature


def step1(project, version, compiled_path):
    coverage_dirname = ProjectConfig.coverage_dirname
    coverage_test2method_dirname = ProjectConfig.coverage_test2method_dirname
    coverage_path = f"{compiled_path}/{coverage_dirname}"
    xml_file_list = os.listdir(coverage_path)
    for xml_file_name in xml_file_list:
        logging.info(f"Analysis coverage for {xml_file_name} in {project}-{version}.")
        methods_list = set()
        xml_file_path = f"{coverage_path}/{xml_file_name}"
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        packages = root.find("packages").findall("package")
        for package in packages:
            classes = package.find("classes").findall("class")
            for clz in classes:
                clz_path = clz.attrib["name"].split("$")[0]
                clz_name = clz_path[clz_path.rindex(".") + 1:]
                clz_filename = clz.attrib["filename"]
                methods = clz.find("methods").findall("method")
                for method in methods:
                    method_name = method.attrib["name"]
                    if method_name == "<clinit>":
                        continue
                    if method_name == "<init>":
                        method_name = clz_name
                    if "access$" in method_name:
                        continue
                    method_name = method_name
                    method_signature = method.attrib["signature"]
                    method_signature = decode_coverage_signature(method_signature, clz_path)
                    hit = method.attrib["line-rate"]
                    if hit != "0.0":
                        methods_list.add(f"{clz_path}${method_name} {method_signature}")
        methods_list = list(methods_list)
        xml_name = xml_file_name[:-4]
        save_dir = f"{compiled_path}/{coverage_test2method_dirname}"
        Utils.create_path(save_dir)
        with open(f"{save_dir}/{xml_name}.txt", "w+", newline="") as f:
            for method_info in methods_list:
                f.write(method_info + os.linesep)
    return True


def run(project, version):
    compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}/{version}"
    Utils.create_path(compiled_path)
    if not step1(project, version, compiled_path):
        logging.error(f"Error at {project}-{version} when running step1.")

if __name__ == '__main__':
    projects = Utils.get_projects()
    for __project in projects:
        versions = Utils.get_active_bug(__project)
        for __version in versions:
            run(__project, __version)
