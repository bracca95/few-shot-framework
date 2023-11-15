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
from src.models.FSL.ProtoNet.proto_loss import ProtoLoss, TestResult, InferenceResult
from src.models.FSL.ProtoNet.proto_extra_modules import ProtoEnhancements
from src.train_test.routine import TrainTest
from src.utils.config_parser import TrainTest as TrainTestConfig
from src.utils.config_parser import Config
from lib.glass_defect_dataset.src.datasets.dataset import DatasetWrapper, DatasetLauncher
from lib.glass_defect_dataset.src.datasets.custom.defectviews import GlassOpt
from lib.glass_defect_dataset.src.utils.tools import Tools, Logger
from lib.glass_defect_dataset.config.consts import General as _CG


class ProtoRoutine(TrainTest):

    W_LOSS = False

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset: DatasetWrapper):
        super().__init__(train_test_config, model, dataset)
        self.learning_rate = 0.001
        self.lr_scheduler_gamma = 0.5
        self.lr_scheduler_step = 20

        # python's linter does not take into account this check, but the rest is correct
        if self._model_config.fsl is None:
            raise ValueError("fsl field cannot be null in config")
        
        # extra modules (if `enhancement` is specified)
        self.mod = ProtoEnhancements(self.model, self._model_config.fsl)

    def init_loader(self, split_set: str):
        current_dataset = getattr(self.dataset_wrapper, f"{split_set}_dataset")
        
        if current_dataset is None:
            return None
        
        if split_set == self.train_str:
            num_samples = self._model_config.fsl.train_k_shot_s + self._model_config.fsl.train_k_shot_q
        else:
            num_samples = self._model_config.fsl.test_k_shot_s + self._model_config.fsl.test_k_shot_q
        
        if any(map(lambda x: x < num_samples, current_dataset.info_dict.values())):
            if split_set == self.val_str:
                Logger.instance().error(f"Skip validation! Val set has not enough elements in some class. Check train s+q config.")
                return None
            raise ValueError(f"At least one class has not enough elements {(num_samples)}. Check {current_dataset.info_dict}")
        
        label_list = current_dataset.label_list
        
        sampler = PrototypicalBatchSampler(
            label_list,
            self._model_config.fsl.train_n_way if split_set == self.train_str else self._model_config.fsl.test_n_way,
            num_samples,
            self._model_config.fsl.episodes
        )
        return DataLoader(current_dataset, batch_sampler=sampler)

    def train(self):
        Logger.instance().debug("Start training")

        trainloader = self.init_loader(self.train_str)
        valloader = self.init_loader(self.val_str)

        optim_param = [{ "params": self.mod.base_model.parameters() }]
        optim_param.append({"params": e.parameters() for e in self.mod.module_list if e is not None})
        optim_param = [o for o in optim_param if o]
        optim = torch.optim.Adam(params=optim_param, lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.lr_scheduler_gamma,
            step_size=self.lr_scheduler_step
        )
        
        train_loss = { }
        train_acc = []
        val_loss = { }
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
        val_config = (fsl_cfg.test_n_way, fsl_cfg.test_k_shot_s, fsl_cfg.test_k_shot_q, fsl_cfg.episodes)

        for epoch in range(self.train_test_config.epochs):
            Logger.instance().debug(f"=== Epoch: {epoch} ===")
            self.mod.train()
            for x, y in tqdm(trainloader):
                optim.zero_grad()
                criterion = ProtoLoss(self.mod, sqrt_eucl=True, tot_epochs=self.train_test_config.epochs, weighted=self.W_LOSS)
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.mod.base_model(x)
                criterion.compute_loss(model_output, target=y, n_way=n_way, n_support=k_support, n_query=k_query, epoch=epoch)
                loss, acc = (criterion.loss, criterion.acc)
                loss.backward()
                optim.step()
                ProtoLoss.append_epoch_loss(train_loss, criterion.loss_dict)
                train_acc.append(acc.item())
            
            avg_loss = { k: np.mean(v[-self._model_config.fsl.episodes:]) for k, v in train_loss.items() }
            avg_acc = np.mean(train_acc[-self._model_config.fsl.episodes:])
            lr_scheduler.step()
            
            Logger.instance().debug(", ".join(f"{k}: {v}" for k, v in avg_loss.items()))
            Logger.instance().debug(f"Avg train accuracy: {avg_acc}")

            # save model
            if avg_acc >= best_acc:
                Logger.instance().debug(f"Found the best model at epoch {epoch}!")
                best_acc = avg_acc
                self.mod.save_models(best_model_path)

            if avg_loss["total_loss"] < best_loss:
                best_loss = avg_loss["total_loss"]
            
            # wandb
            wdb_dict = { "train_loss": avg_loss["total_loss"], "train_acc": avg_acc }

            ## VALIDATION
            if valloader is not None:
                avg_loss_eval, avg_acc_eval = self.validate(val_config, valloader, epoch, val_loss, val_acc)
                if avg_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    self.mod.save_models(val_model_path)

                # wandb
                wdb_dict["val_loss"] = avg_loss_eval
                wdb_dict["val_acc"] = avg_acc_eval    
            ## EOF: VALIDATION
            
            # wandb
            wandb.log(wdb_dict)

            # stop conditions and save last model
            if epoch == self.train_test_config.epochs-1 or self.check_stop_conditions(avg_loss["total_loss"], best_acc):
                pth_path = last_val_model_path if valloader is not None else last_model_path
                Logger.instance().debug(f"STOP: saving last epoch model named `{os.path.basename(pth_path)}`")
                self.mod.save_models(pth_path)

                # wandb: save all models
                wandb.save(f"{out_folder}/best_model.pth", base_path=os.getcwd())

                return

    def validate(self, val_config: Tuple, valloader: DataLoader, epoch: int, val_loss: dict, val_acc: List[float]):
        Logger.instance().debug("Validating!")

        n_way, k_support, k_query, episodes = (val_config)
        
        self.mod.eval()
        with torch.no_grad():
            for x, y in valloader:
                criterion = ProtoLoss(self.mod, sqrt_eucl=True, tot_epochs=self.train_test_config.epochs, weighted=self.W_LOSS)
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.mod.base_model(x)
                criterion.compute_loss(model_output, target=y, n_way=n_way, n_support=k_support, n_query=k_query, epoch=epoch)
                _, acc = (criterion.loss, criterion.acc)
                ProtoLoss.append_epoch_loss(val_loss, criterion.loss_dict)
                val_acc.append(acc.item())
            avg_loss_eval = { k: np.mean(v[-episodes:]) for k, v in val_loss.items() }
            avg_acc_eval = np.mean(val_acc[-episodes:])

        Logger.instance().debug(", ".join(f"{k}: {v}" for k, v in avg_loss_eval.items()))
        Logger.instance().debug(f"Avg validation accuracy: {avg_acc_eval}")

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

        self.mod.load_models(model_path)
        
        legacy_avg_acc = list()
        acc_per_epoch = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.keys())) }

        tr_acc_max = 0.0
        tr_max = TestResult()

        n_way, k_support, k_query = (self._model_config.fsl.test_n_way, self._model_config.fsl.test_k_shot_s, self._model_config.fsl.test_k_shot_q)
        
        self.mod.eval()
        with torch.no_grad():
            for epoch in tqdm(range(10)):
                tr = TestResult()
                score_per_class = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.keys())) }
                for x, y in testloader:
                    x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                    y_pred = self.mod.base_model(x)

                    # (overall accuracy [legacy], accuracy per class)
                    legacy_acc, acc_vals = tr.proto_test(y_pred, target=y, n_way=n_way, n_support=k_support, n_query=k_query, enhance=self.mod, sqrt_eucl=True)
                    legacy_avg_acc.append(legacy_acc.item())
                    for k, v in acc_vals.items():
                        score_per_class[k] = torch.cat((score_per_class[k], v.reshape(1,)))
                
                avg_score_class = { k: torch.mean(v) for k, v in score_per_class.items() }
                avg_score_class_print = { k: v.item() for (k, v) in zip(self.test_info.keys(), avg_score_class.values()) }
                Logger.instance().debug(f"at epoch {epoch}, average test accuracy: {avg_score_class_print}")

                for k, v in avg_score_class.items():
                    acc_per_epoch[k] = torch.cat((acc_per_epoch[k], v.reshape(1,)))

                tr.acc_overall = tr.acc_overall.mean()
                if tr.acc_overall > tr_acc_max:
                    tr_max = tr

        avg_acc_epoch = { k: torch.mean(v) for k, v in acc_per_epoch.items() }
        avg_acc_epoch_print = { k: v.item() for (k, v) in zip(self.test_info.keys(), avg_acc_epoch.values()) }
        Logger.instance().debug(f"Accuracy on epochs: {avg_acc_epoch_print}")
        
        legacy_avg_acc = np.mean(legacy_avg_acc)
        Logger.instance().debug(f"Legacy test accuracy: {legacy_avg_acc}")

        if self._model_config.fsl.test_n_way == len(self.test_info.keys()) and len(tr_max.target_inds) != 0 and len(tr_max.y_hat) != 0:
            y_true = tr_max.target_inds
            preds = tr_max.y_hat
            wandb.log({
                "confusion": wandb.plot.confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    preds=preds.cpu().detach().numpy(),
                    class_names=list(self.dataset_wrapper.label_to_idx.keys())
                    )
                })
            

class ProtoInference:

    SUPPORT = "support"
    QUERY = "query"

    def __init__(self, config: Config, model: Model, support_set: GlassOpt, query_set: GlassOpt):
        self.config = config
        self.model = model
        self.support_set = support_set
        self.query_set = query_set

        if self.config.model.fsl is None:
            raise ValueError(f"In config/config.json file, `fsl` cannot be Null")
        self.mod = ProtoEnhancements(self.model, self.config.model.fsl)
        self.n_way = self.config.model.fsl.test_n_way
        self.k_shot_s = self.config.model.fsl.test_k_shot_s
        self.k_shot_q = self.config.model.fsl.test_k_shot_q

    def _init_support_loader(self) -> DataLoader:
        imgs_in_dir = list(filter(lambda x: x.endswith(".png"), os.listdir(os.path.join(self.config.dataset.dataset_path, self.SUPPORT))))

        # check the number of images is correct
        if not len(imgs_in_dir) == (2 * self.n_way * self.k_shot_s):
            raise ValueError(
                f"There must be {self.n_way} * {self.k_shot_s} ({self.n_way * self.k_shot_s}) defects in the directory. " +
                f"The current directory contains {len(imgs_in_dir)} images, instead. " +
                f"IMPORTANT: If you are using two channels, the number of expected files duplicates."
            )

        # check that there are k_shot for every class
        defect_names = list(map(lambda x: x.rsplit("_did", 1)[0], imgs_in_dir))
        defect_classes = set(defect_names)
        numel_defects_per_class = [defect_names.count(c) for c in defect_classes]
        if any(map(lambda x: not x == ( 2 * self.k_shot_s), numel_defects_per_class)):
            raise ValueError(
                f"Every class ({defect_classes}) must contain exactly {self.k_shot_s} defects, " +
                f"i.e. {2 * self.k_shot_s} images if two channels are required."
            )
        
        return DataLoader(self.support_set.test_dataset, batch_size=(self.n_way * self.k_shot_s), collate_fn=self.custom_collate_fn)
    
    def _init_query_loader(self, _batch_size: Optional[int]=None):
        batch_size = self.n_way * self.k_shot_q

        if _batch_size is not None:
            batch_size = _batch_size

        Logger.instance().debug(f"Batch size for queries has been set to {batch_size}.")

        image_batch = []
        label_batch = []
        path_batch = []

        # CHECK could shuffle, but for inference in standard protonet is not necessary
        # use yield to return ALL the query images
        for img, lbl, path in self.query_set.test_dataset:
            image_batch.append(img)
            label_batch.append(lbl)
            path_batch.append(path)
            if len(image_batch) == batch_size:
                yield torch.stack(image_batch, dim=0), torch.LongTensor(label_batch), path_batch
                image_batch, label_batch, path_batch = [], [], []

        # If there are any remaining images in image_batch after the loop, yield them
        if image_batch:
            yield torch.stack(image_batch, dim=0), torch.LongTensor(label_batch), path_batch

    def test(self, model_path: str):
        Logger.instance().debug("Start inference")

        try:
            model_path = Tools.validate_path(model_path)
            self.mod.load_models(model_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)
        except ValueError as ve:
            Logger.instance().error(f"{ve.args}. No test performed")
            return

        support_loader = self._init_support_loader()
        query_loader = self._init_query_loader()

        self.mod.eval()
        with torch.no_grad():
            # load support batch (always the same)
            sx, sy, _ = next(iter(support_loader))
            sx, sy = sx.to(_CG.DEVICE), sy.to(_CG.DEVICE)
            s_emb = self.mod.base_model(sx)
            dim = s_emb.size(-1)

            ir = InferenceResult(self.mod, s_emb, self.n_way, self.k_shot_s, dim, self.support_set.idx_to_label, True)
            ir.proto_inference(query_loader)
            acc_overall, acc_vals, recall, precision = ir.compute_accuracy()

            Logger.instance().debug(
                f"overall accuracy: {acc_overall}\n" +
                f"accuracy values (class): {acc_vals}\n" +
                f"recall: {recall}\n" +
                f"precision: {precision}\n"
            )

        wandb.save(f"{os.path.join(os.getcwd(), 'output/log.log')}", base_path=os.getcwd())
            
    @staticmethod
    def custom_collate_fn(batch):
        sorted_batch = sorted(batch, key=lambda x: x[1])
        images, labels, paths = zip(*sorted_batch)

        return torch.stack(images, dim=0), torch.LongTensor(labels), paths