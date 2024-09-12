import csv
import logging
import os
import sys
import time
import traceback

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
import concurrent.futures

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
    from dataset.dataset import DictCustomDataset, dic_collate_fn
    from model import TrainModel
except ImportError:
    from ..config import ProjectConfig
    from ..defects4j.utils import Utils
    from ..dataset.dataset import DictCustomDataset, dic_collate_fn
    from ..model.model import TrainModel


def evaluate_single_version(project, model, test_version, save_path, cuda_device):
    try:
        if os.path.exists(f"{save_path}/{test_version}.csv"):
            logging.info(f"{save_path}/{test_version}.csv is existed. skip.")
            return
        model.cuda(cuda_device)
        model.eval()
        with torch.no_grad():
            project_compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}"
            _dataset = DictCustomDataset(project_compiled_path, selected_versions=[test_version],
                                         use_resample=False)
            val_loader = DataLoader(_dataset, batch_size=batch_size, collate_fn=dic_collate_fn,
                                    num_workers=0)
            result = []
            for passed_test_cases, failed_test_cases, history_change, call_information, methods, labels in tqdm(
                    val_loader):
                x = [t.cuda(cuda_device) for t in
                     [passed_test_cases, failed_test_cases, history_change, call_information, methods]]
                preds = model(x[0], x[1], x[2], x[3], x[4]).squeeze(-1).detach().cpu()
                for label, pred in zip(labels, preds):
                    result.append((int(label.item()), pred.item()))
            result = sorted(result, key=lambda t: t[1], reverse=True)
            with open(os.path.join(save_path, f"{test_version}.csv"),
                      "w+") as f:
                cf = csv.writer(f)
                cf.writerow(("Label", "Predict"))
                cf.writerows(result)
    except:
        logging.error(traceback.format_exc())


def calc_all_version_metrics(save_path, project):
    for test_version in Utils.get_active_bug(project):
        if not os.path.exists(f"{save_path}/{test_version}.csv"):
            logging.info(f"{save_path}/{test_version}.csv is not existed. continue.")
            continue
        with open(f"{save_path}/{test_version}.csv", "r") as f:
            rf = csv.reader(f)
            next(rf)
            result = list()
            for item in rf:
                result.append((int(item[0]), float(item[1])))
            top1, top3, top5, FR, AR = calc(result)
        with open(os.path.join(save_path, f"{test_version}-metrics.csv"), "w+") as f:
            cf = csv.writer(f)
            cf.writerow(("top1", "top3", "top5", "FR", "AR"))
            cf.writerow((top1, top3, top5, FR, AR))


def calc(data):
    top1 = 0
    top3 = 0
    top5 = 0
    AR = list()
    FR = 0
    for i, item in enumerate(data):
        label, predict = item
        if i < 1:
            if label == 1:
                top1 += 1
                top3 += 1
                top5 += 1
        elif i < 3:
            if label == 1:
                top3 += 1
                top5 += 1
        elif i < 5:
            if label == 1:
                top5 += 1
        if label == 1:
            if FR == 0:
                FR = (i + 1)
            AR.append(i + 1)
    AR = mean(AR)
    return top1, top3, top5, FR, AR


def mean(data):
    if len(data) != 0:
        return sum(data) / len(data)


def merge_all(save_path):
    top1s = list()
    top3s = list()
    top5s = list()
    MFR = list()
    MAR = list()
    for test_version in tqdm(active_bug):
        matric_file = f"{save_path}/{test_version}-metrics.csv"
        if not os.path.exists(matric_file):
            logging.info(f"{matric_file} is not existed. continue.")
            continue
        with open(matric_file, "r") as f:
            rf = csv.reader(f)
            next(rf)
            top1, top3, top5, FR, AR = [t if t != "" else None for t in next(rf)]
            top1, top3, top5 = int(top1), int(top3), int(top5)
            FR = float(FR) if FR is not None else None
            AR = float(AR) if AR is not None else None
            top3 = int((top3 > 0))
            top5 = int((top5 > 0))
            top1s.append(top1)
            top3s.append(top3)
            top5s.append(top5)
            MFR.append(FR)
            MAR.append(AR)
    num_samples = len(top1s)
    top1s = sum(top1s)
    top3s = sum(top3s)
    top5s = sum(top5s)
    MFR = mean(MFR)
    MAR = mean(MAR)
    with open(f"{save_path}/val_metrics.csv", "w+") as f:
        cf = csv.writer(f)
        cf.writerow(("top1", "top3", "top5", "MAR", "MFR", "num_samples"))
        cf.writerow((top1s, top3s, top5s, MAR, MFR, num_samples))


class MyEarlyStopping(Callback):
    def __init__(self):
        self.last_loss = 9999

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        best_FR = trainer.callback_metrics.get("best_FR")
        if best_FR == 1:
            current_loss = trainer.callback_metrics.get("val_loss")
            if self.last_loss < current_loss:
                logging.info(f"Early stopped for best_FR={best_FR} val_loss={current_loss}.")
                trainer.should_stop = True
            else:
                self.last_loss = current_loss


def train(project, time_str, selected_versions, test_version, cuda_devices):
    try:
        logging.info(f"Training {project}_{time_str}_TestVersion{test_version} in cuda-{cuda_devices}.")
        project_compiled_path = f"{ProjectConfig.path_dataset_home}/compiled/{project}"
        _val_dataset = DictCustomDataset(project_compiled_path, selected_versions=[test_version],
                                         use_resample=False, version_cache=version_cache)
        if len(_val_dataset.dataset_label_true) < 1:
            logging.info(
                f"No bug method in version {test_version}.")
            return
        _dataset = DictCustomDataset(project_compiled_path, selected_versions=selected_versions,
                                     use_resample=True, version_cache=version_cache)
        train_loader = DataLoader(_dataset, batch_size=batch_size, collate_fn=dic_collate_fn, shuffle=True,
                                  num_workers=num_worker)
        val_loader = DataLoader(_val_dataset, batch_size=batch_size, collate_fn=dic_collate_fn,
                                num_workers=num_worker)
        logger = TensorBoardLogger(f'HetFL_Model/', name=f"{project}_{time_str}",
                                   version=f"{test_version}", log_graph=False)
        model = TrainModel()
        model_checkpoint = ModelCheckpoint(monitor='score', filename="best")
        early_stopping = MyEarlyStopping()
        max_epochs = 1
        val_check_interval = 1 / 20
        trainer = Trainer(accelerator="cuda", devices=cuda_devices, max_epochs=max_epochs, logger=logger,
                          log_every_n_steps=10, val_check_interval=val_check_interval,
                          callbacks=[model_checkpoint, early_stopping])
        trainer.fit(model, train_loader, val_loader)
    except:
        logging.error(traceback.format_exc())
        exit()


batch_size = 32
n = 1
num_worker = 0
cuda_device = 3
evaluate_save_root_path = "model_output"
if __name__ == '__main__':
    t = False
    start_time = str(int(time.time()))
    for iii in range(n):
        time_str = str(iii) + "TT" + start_time
        switch_index = 0
        for project in Utils.get_projects():
            if project in ["Lang"]:
                batch_size = 16
            else:
                batch_size = 32
            active_bug = Utils.get_active_bug(project)
            active_bug_set = set(active_bug)
            model_root_path = f"HetFL_Model/{project}_{time_str}"
            evaluate_save_path = f"{evaluate_save_root_path}/{project}_{time_str}"
            if not os.path.exists(evaluate_save_path):
                os.makedirs(evaluate_save_path)
            for _i, test_version in enumerate(active_bug):
                model_path = f"{model_root_path}/{test_version}/checkpoints/best.ckpt"
                selected_versions = list(active_bug_set - {test_version})
                train(project, time_str, selected_versions, test_version, [cuda_device])
                if os.path.exists(
                        f"HetFL_Model/{project}_{time_str}/{test_version}/checkpoints/best.ckpt"):
                    __model = TrainModel.load_from_checkpoint(
                        f"HetFL_Model/{project}_{time_str}/{test_version}/checkpoints/best.ckpt")
                    logging.info(f"Evaluating {project}_{time_str}_TestVersion{test_version} in cuda-{cuda_device}.")
                    evaluate_single_version(project, __model, test_version, evaluate_save_path, cuda_device)
                    os.system(f"rm -rf {model_root_path}/{test_version}")
                switch_index = switch_index + 1
            os.system(f"rm -rf {model_root_path}")
            calc_all_version_metrics(evaluate_save_path, project)
            merge_all(evaluate_save_path)
        t = True
