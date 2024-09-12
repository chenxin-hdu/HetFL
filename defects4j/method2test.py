import logging
import os
import pickle
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
except ImportError:
    from ..config import ProjectConfig
    from .utils import Utils


def step1(project, version, work_path, compiled_path):
    try:
        logging.info(f"Load test2method in {project}-{version}.")
        test_short_name_reflect = dict()
        with open(f"{compiled_path}/{ProjectConfig.all_tests_filename}", "r") as f:
            while (line := f.readline()) != "":
                shortname = line.split("$")[-1].strip()
                if test_short_name_reflect.get(shortname) is None:
                    test_short_name_reflect[shortname] = line.strip()
        method2test = dict()
        for file in os.listdir(f"{compiled_path}/{ProjectConfig.coverage_test2method_dirname}"):
            shortname = file[:-4]
            long_name = test_short_name_reflect.get(shortname)
            if long_name is None:
                continue
            filepath = f"{compiled_path}/{ProjectConfig.coverage_test2method_dirname}/{file}"
            with open(filepath, "r") as f:
                while (line := f.readline()) != "":
                    method = line.strip()
                    if method2test.get(method) is None:
                        method2test[method] = [long_name]
                    else:
                        method2test[method] = method2test[method] + [long_name]
        with open(f"{compiled_path}/{ProjectConfig.method2test_filename}", "wb+") as f:
            pickle.dump(method2test, f)
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
