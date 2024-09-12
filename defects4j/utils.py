import os
import sys

import pandas

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

from config import ProjectConfig


class __Utils:
    def get_active_bug(self, project):
        path = f"{ProjectConfig.path_defects4j}/framework/projects/{project}/active-bugs.csv"
        data = pandas.read_csv(path)
        data = list(data['bug.id'])
        if project == "Closure":
            data = [t for t in data if t <= 133]
        return data

    def get_projects(self):
        return ["Chart", "Lang", "Math", "Time", "Closure", "Cli", "Codec", "Collections", "Compress", "Csv", "Gson",
                "JacksonCore", "JacksonXml", "Jsoup", "JxPath", "JacksonDatabind",
                "Mockito"]

    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_method_signature(self, method_node):
        if hasattr(method_node, "parameters"):
            method_param_signatures = list()
            for param in method_node.parameters:
                signature = ""
                for t in range(len(param.type.dimensions)):
                    signature += "["
                # signature += self.try_replace_by_dict(param.type.name, dict_import)
                if len(param.type.name) > 1:
                    signature += param.type.name
                else:
                    signature += "Object"
                method_param_signatures.append(signature)
        else:
            method_param_signatures = ""
        if len(method_param_signatures) > 0:
            method_method_signature = ";".join(method_param_signatures)
        else:
            method_method_signature = ""
        if not hasattr(method_node, "return_type") or method_node.return_type is None:
            method_return_name = "void"
        else:
            signature = ""
            for t in range(len(method_node.return_type.dimensions)):
                signature += "["
            if len(method_node.return_type.name) > 1:
                signature += method_node.return_type.name
            else:
                signature += "Object"
            method_return_name = signature
        # method_return_signature = self.try_replace_by_dict(method_return_name, dict_import)
        method_return_signature = method_return_name
        return f'({method_method_signature}){method_return_signature}'

    def try_replace_by_dict(self, value, _dict: dict):
        if r := _dict.get(value):
            return r
        else:
            return value


Utils = __Utils()
