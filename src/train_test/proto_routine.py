import os
import sys
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple

from src.models.model import Model
from src.models.FSL.ProtoNet.proto_batch_sampler import PrototypicalBatchSampler
from src.models.FSL.ProtoNet.proto_loss import ProtoTools, TestResult
from src.train_test.routine import TrainTest
from src.utils.tools import Tools, Logger
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.glass_defect_dataset.src.datasets.dataset import CustomDataset
from lib.glass_defect_dataset.config.consts import General as _CG


class ProtoRoutine(TrainTest):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset: CustomDataset):
        super().__init__(train_test_config, model, dataset)
        self.learning_rate = 0.001
        self.lr_scheduler_gamma = 0.5
        self.lr_scheduler_step = 20

        # python's linter does not take into account this check, but the rest is correct
        if self._model_config.fsl is None:
            raise ValueError("fsl field cannot be null in config")

    def init_loader(self, split_set: str):
        current_subset = self.dataset.get_subset_info(split_set)
        
        if current_subset.subset is None:
            return None
        
        min_req = self._model_config.fsl.test_k_shot_q + self._model_config.fsl.train_k_shot_s
        if any(map(lambda x: x < min_req, current_subset.info_dict.values())):
            raise ValueError(f"at least one class has not enough elements {(min_req)}. Check {current_subset.info_dict}")
        
        label_list = [self.dataset[idx][1] for idx in current_subset.subset.indices]
        if split_set == self.train_str:
            num_samples = self._model_config.fsl.train_k_shot_s + self._model_config.fsl.train_k_shot_q
        else:
            num_samples = self._model_config.fsl.test_k_shot_s + self._model_config.fsl.test_k_shot_q
        
        sampler = PrototypicalBatchSampler(
            label_list,
            self._model_config.fsl.train_n_way if split_set == self.train_str else self._model_config.fsl.test_n_way,
            num_samples,
            self._model_config.fsl.episodes
        )
        return DataLoader(current_subset.subset, batch_sampler=sampler)

    def train(self):
        Logger.instance().debug("Start training")

        trainloader = self.init_loader(self.train_str)
        valloader = self.init_loader(self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.lr_scheduler_gamma,
            step_size=self.lr_scheduler_step
        )
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc: float = 0.0
        best_loss = float("inf")

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        best_model_path = os.path.join(out_folder, "best_model.pth")
        val_model_path = os.path.join(out_folder, "val_model.pth")
        last_model_path = os.path.join(out_folder, "last_model.pth")
        last_val_model_path = os.path.join(out_folder, "last_val_model.pth")

        fsl_cfg = self._model_config.fsl
        n_way, k_support, k_query = (fsl_cfg.train_n_way, fsl_cfg.train_k_shot_s, fsl_cfg.train_k_shot_q)
        val_config = (fsl_cfg.train_n_way, fsl_cfg.train_k_shot_s, fsl_cfg.train_k_shot_q, fsl_cfg.episodes)

        for eidx, epoch in enumerate(range(self.train_test_config.epochs)):
            Logger.instance().debug(f"=== Epoch: {epoch} ===")
            self.model.train()
            self.mod.train()
            for x, y in tqdm(trainloader):
                optim.zero_grad()
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                
                model_output = self.model(x)
                s_batch, q_batch = ProtoTools.split_support_query(model_output, target=y, n_way=n_way, n_support=k_support, n_query=k_query)
                loss, acc = ProtoTools.proto_loss(s_batch, q_batch)
                
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            
            avg_loss = np.mean(train_loss[-self._model_config.fsl.episodes:])
            avg_acc = np.mean(train_acc[-self._model_config.fsl.episodes:])
            lr_scheduler.step()
            
            Logger.instance().debug(f"Avg Train Loss: {avg_loss}, Avg Train Acc: {avg_acc}")

            # save model
            if avg_acc >= best_acc:
                Logger.instance().debug(f"Found the best model at epoch {epoch}!")
                best_acc = avg_acc
                torch.save(self.model.state_dict(), best_model_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # wandb
            wdb_dict = { "train_loss": avg_loss, "train_acc": avg_acc }

            ## VALIDATION
            if valloader is not None:
                avg_loss_eval, avg_acc_eval = self.validate(val_config, valloader, val_loss, val_acc)
                if avg_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), val_model_path)

                # wandb
                wdb_dict["val_loss"] = avg_loss_eval
                wdb_dict["val_acc"] = avg_acc_eval    
            ## EOF: VALIDATION
            
            # wandb
            wandb.log(wdb_dict)

            # stop conditions and save last model
            if eidx == self.train_test_config.epochs-1 or self.check_stop_conditions(best_acc):
                pth_path = last_val_model_path if valloader is not None else last_model_path
                Logger.instance().debug(f"STOP: saving last epoch model named `{os.path.basename(pth_path)}`")
                torch.save(self.model.state_dict(), pth_path)

                # wandb: save all models
                wandb.save(f"{out_folder}/*.pth")

                return

    def validate(self, val_config: Tuple, valloader: DataLoader, val_loss: List[float], val_acc: List[float]):
        Logger.instance().debug("Validating!")

        n_way, k_support, k_query, episodes = (val_config)
        
        self.model.eval()
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                s_batch, q_batch = ProtoTools.split_support_query(model_output, target=y, n_way=n_way, n_support=k_support, n_query=k_query)
                loss, acc = ProtoTools.proto_loss(s_batch, q_batch)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            avg_loss_eval = np.mean(val_loss[-episodes:])
            avg_acc_eval = np.mean(val_acc[-episodes:])

        Logger.instance().debug(f"Avg Val Loss: {avg_loss_eval}, Avg Val Acc: {avg_acc_eval}")

        return avg_loss_eval, avg_acc_eval

    def test(self, model_path: str):
        Logger.instance().debug("Start testing")
        
        if self._model_config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")
        
        try:
            model_path = Tools.validate_path(model_path)
            testloader = self.init_loader(self.test_str)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)
        except ValueError as ve:
            Logger.instance().error(f"{ve.args}. No test performed")
            return

        self.model.load_state_dict(torch.load(model_path))
        
        legacy_avg_acc = list()
        acc_per_epoch = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.info_dict.keys())) }

        tr_acc_max = 0.0
        tr_max = TestResult()

        n_way, k_support, k_query = (self._model_config.fsl.test_n_way, self._model_config.fsl.test_k_shot_s, self._model_config.fsl.test_k_shot_q)
        
        self.model.eval()
        with torch.no_grad():
            for epoch in tqdm(range(10)):
                tr = TestResult()
                score_per_class = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.info_dict.keys())) }
                for x, y in testloader:
                    x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                    y_pred = self.model(x)
                    s_batch, q_batch = ProtoTools.split_support_query(y_pred, target=y, n_way=n_way, n_support=k_support, n_query=k_query)

                    # (overall accuracy [legacy], accuracy per class)
                    legacy_acc, acc_vals = tr.proto_test(s_batch, q_batch)
                    legacy_avg_acc.append(legacy_acc.item())
                    for k, v in acc_vals.items():
                        score_per_class[k] = torch.cat((score_per_class[k], v.reshape(1,)))
                
                avg_score_class = { k: torch.mean(v) for k, v in score_per_class.items() }
                avg_score_class_print = { k: v.item() for (k, v) in zip(self.test_info.info_dict.keys(), avg_score_class.values()) }
                Logger.instance().debug(f"at epoch {epoch}, average test accuracy: {avg_score_class_print}")

                for k, v in avg_score_class.items():
                    acc_per_epoch[k] = torch.cat((acc_per_epoch[k], v.reshape(1,)))

                tr.acc_overall = tr.acc_overall.mean()
                if tr.acc_overall > tr_acc_max:
                    tr_max = tr

        avg_acc_epoch = { k: torch.mean(v) for k, v in acc_per_epoch.items() }
        avg_acc_epoch_print = { k: v.item() for (k, v) in zip(self.test_info.info_dict.keys(), avg_acc_epoch.values()) }
        Logger.instance().debug(f"Accuracy on epochs: {avg_acc_epoch_print}")
        
        legacy_avg_acc = np.mean(legacy_avg_acc)
        Logger.instance().debug(f"Legacy test accuracy: {legacy_avg_acc}")

        if self._model_config.fsl.test_n_way == len(self.dataset.label_to_idx.keys()) and len(tr_max.target_inds) != 0 and len(tr_max.y_hat) != 0:
            y_true = tr_max.target_inds
            preds = tr_max.y_hat
            wandb.log({
                "confusion": wandb.plot.confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    preds=preds.cpu().detach().numpy(),
                    class_names=list(self.dataset.label_to_idx.keys())
                    )
                })
