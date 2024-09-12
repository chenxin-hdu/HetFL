import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d:%(filename)s:%(lineno)d:%(levelname)s:\t%(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S', )


class __ProjectConfig:
    def __init__(self):
        self.path_home = None # work home path
        self.path_defects4j = f"{self.path_home}/defects4j"
        self.path_dataset_home = f"{self.path_home}/dataset/defects4j"
        self.coverage_dirname = "coverage"
        self.java_callgraph2_origin_dir_path = None # java-callgraph2 path
        self.java_callgraph2_dirname = "java-callgraph2"
        self.coverage_test2method_dirname = f"{self.coverage_dirname}_test2method"
        self.method2test_filename = f"method2test.pkl"
        self.coverage_failing_dirname = f"{self.coverage_dirname}_failing_stacktrace"
        self.classes_path_filename = f"classes_path.txt"
        self.raw_method_attributes_filename = f"raw_method_attributes.pkl"
        self.raw_test_attributes_filename = f"raw_test_attributes.pkl"
        self.methods_history_filename = f"methods_history.pkl"
        self.classes_relevant_filename = "classes_relevant.txt"
        self.raw_tests_trigger_filename = "raw_tests_trigger"
        self.tests_trigger_filename = "tests_trigger.txt"
        self.classes_modified_filename = "classes_modified.txt"
        self.source_root_dir_file = "source_root.txt"
        self.test_root_dir_file = "test_root.txt"
        self.pass_tests_filename = "passed_tests.txt"
        self.failing_tests_filename = "failing_tests.txt"
        self.all_tests_filename = "all_tests.txt"
        self.git_log_filename = "git_log.txt"
        self.git_diff_buggy_fix_filename = "git_diff_buggy_fix.txt"
        self.bug_method_filename = "bug_method.txt"
        self.raw_method_call_filename = "raw_method_call.txt"
        self.method_call_filename = "method_call.pkl"
        self.vector_method_name_filename = "vector_method_call.pkl"
        self.vector_method_attribute_filename = "vector_method_attribute.pkl"
        self.vector_method_call_filename = "vector_method_call.pkl"
        self.vector_pass_test_attribute_filename = "vector_pass_test_attribute.pkl"
        self.vector_fail_test_attribute_filename = "vector_fail_test_attribute.pkl"
        self.vector_test_content_filename = "vector_test_content.pkl"
        self.vector_method_history_filename = "vector_method_history.pkl"
        self.relevant_method_list = "relevant_method_list.txt"
        self.all_method_list = "all_method_list.txt"
        self.project_history_temp_dirname = "history_tmp"
        self.method_call_temp_dirname = "method_call_tmp"
        self.version_tokenizer_train_file = "version_tokenizer_train_file.txt"
        self.pretrain_tokenizer_path = None # "tokenizer.json"
        self.labels_filename = "labels.pkl"
        self.word2vec_train_data_dir = "word2vec_train_data"
        self.word2vec = "word2vec"
        self.word2vec_model = "word2vec.model"
        self.node2vec_model = "node2vec.model"
        self.tmp_path = None # dirpath

    def create_dirs(self):
        for attr in dir(self):
            if attr.startswith('path'):
                if os.path.isdir(attr):
                    os.makedirs(attr)
                else:
                    logging.info(f"path {attr} is not a dir. skip.")


ProjectConfig = __ProjectConfig()
