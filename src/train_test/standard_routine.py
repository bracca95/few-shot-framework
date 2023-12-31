import os
import sys
import torch
import wandb

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.models.model import Model
from src.train_test.routine import TrainTest
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.glass_defect_dataset.src.datasets.dataset import DatasetWrapper
from lib.glass_defect_dataset.config.consts import General as _CG
from lib.glass_defect_dataset.src.utils.tools import Logger, Tools


class StandardRoutine(TrainTest):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset: DatasetWrapper):
        super().__init__(train_test_config, model, dataset)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(_CG.DEVICE)

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
        top_pred = y_pred.argmax(1, keepdim=True)           # select the max class (the one with the highest score)
        correct = top_pred.eq(y.view_as(top_pred)).sum()    # count the number of correct predictions
        return correct.float() / y.shape[0]                 # compute percentage of correct predictions (accuracy score)

    @staticmethod
    def class_accuracies(cls_acc_dict: dict, batch_labels: torch.Tensor, batch_pred_idx: torch.Tensor) -> dict:
        for cls_label in torch.unique(batch_labels):
            mask = batch_labels == cls_label
            tot_samples = mask.sum().item()
            correct_preds = (batch_pred_idx[mask] == batch_labels[mask]).sum().item()
            # accuracy = correct_preds / tot_samples  # if tot_samples > 0 else 0.0 # tot_samples cannot be 0

            # sum existing values
            c = cls_label.item()
            cls_acc_dict[c] = [cls_acc_dict[c][0] + correct_preds, cls_acc_dict[c][1] + tot_samples]

        return cls_acc_dict
    
    def init_loader(self, split_set: str):
        current_subset = self.dataset_wrapper.get_subset_info(split_set)
        if current_subset.subset is None:
            return None
        
        return DataLoader(current_subset.subset, batch_size=self.train_test_config.batch_size)

    def train(self):
        Logger.instance().debug("Start training")
        
        trainloader = self.init_loader(self.train_str)
        valloader = self.init_loader(self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        
        # # wandb
        # example_data, examples_target = next(iter(trainloader))
        # img_grid = make_grid(example_data)
        # wandb.log({"examples": wandb.Image(img_grid, caption="Top: Output, Bottom: Input")})

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        best_model_path = os.path.join(out_folder, "best_model.pth")
        val_model_path = os.path.join(out_folder, "val_model.pth")
        last_model_path = os.path.join(out_folder, "last_model.pth")
        last_val_model_path = os.path.join(out_folder, "last_val_model.pth")
        
        best_loss = float('inf')
        best_acc = 0

        for eidx, epoch in enumerate(range(self.train_test_config.epochs)):
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0

            for image, label in tqdm(trainloader, desc="Training", leave=False): 
                optim.zero_grad()
                image = image.to(_CG.DEVICE) #.reshape(-1, shape, shape) # alternative x.view(x.shape[0], -1) -> [batch, dim*dim]
                label = label.to(_CG.DEVICE)

                # forward pass
                pred = self.model(image)                        # [batch_size, n_classes]
                loss = self.criterion(pred, label)
                acc = StandardRoutine.compute_accuracy(pred.data, label)

                # backward and optimize
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_loss = epoch_loss / len(trainloader)
            epoch_acc = epoch_acc / len(trainloader)

            Logger.instance().debug("=================================================================")
            Logger.instance().debug(f"Epoch {eidx} :: loss: {epoch_loss}, accuracy: {epoch_acc}")

            if epoch_acc > best_acc and epoch > 0:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), best_model_path)
                Logger.instance().debug(f"saving model at iteration {epoch}.")

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            # wandb
            wdb_dict = { "train_loss": epoch_loss, "train_acc": epoch_acc }

            ## VALIDATION
            if valloader is not None:
                epoch_loss_eval, epoch_acc_eval = self.validate(valloader)
                if epoch_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), val_model_path)

                # wandb
                wdb_dict["val_loss"] = epoch_loss_eval
                wdb_dict["val_acc"] = epoch_acc_eval
            ## EOF: VALIDATION

            # wandb
            wandb.log(wdb_dict)

            # stop conditions and save last model
            if eidx == self.train_test_config.epochs-1 or self.check_stop_conditions(best_acc):
                pth_path = last_val_model_path if valloader is not None else last_model_path
                Logger.instance().debug(f"STOP: saving last epoch model named `{os.path.basename(pth_path)}`")
                torch.save(self.model.state_dict(), pth_path)

                # wandb: save model(s)
                wandb.save(f"{out_folder}/*.pth", base_path=os.getcwd())

                return

    def validate(self, valloader: DataLoader):
        Logger.instance().debug("Validating!")
        
        tot_samples = 0
        tot_correct = 0

        running_loss = 0.0
        
        self.model.eval()
        for images, labels in valloader:
            images = images.to(_CG.DEVICE)
            labels = labels.to(_CG.DEVICE)
            
            y_pred = self.model(images)
            loss = self.criterion(y_pred, labels)
            
            # max returns (value, index)
            top_pred_val, top_pred_idx = torch.max(y_pred.data, 1)
            n_correct = top_pred_idx.eq(labels.view_as(top_pred_idx)).sum()
            
            # accuracy
            tot_samples += labels.size(0)
            tot_correct += n_correct
            running_loss += loss.item()

        acc = tot_correct / tot_samples
        running_loss /= len(valloader)

        Logger.instance().debug(f"Avg Val Loss: {running_loss}, Avg Val Acc: {acc}")

        return running_loss, acc
    
    def test(self, model_path: str):
        Logger.instance().debug("Start testing")
        
        try:
            model_path = Tools.validate_path(model_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)

        self.model.load_state_dict(torch.load(model_path))
        testloader = self.init_loader(self.test_str)

        # load labels for class accuracies
        current_subset = self.dataset_wrapper.subsets_dict[self.test_str]
        idxs = torch.LongTensor(current_subset.indices)
        label_list = torch.IntTensor(current_subset.dataset.label_list)[idxs]
        classes = torch.unique(label_list)
        
        self.model.eval()
        with torch.no_grad():
            tot_samples = 0
            tot_correct = 0
            all_labels = torch.Tensor([]).to(_CG.DEVICE)
            all_cls_idxs = torch.Tensor([]).to(_CG.DEVICE)
            all_pred_vals = torch.Tensor([]).to(_CG.DEVICE)
            cls_acc_dict = { c.item(): [0, 0] for c in classes }
            for images, labels in testloader:
                images = images.to(_CG.DEVICE)
                labels = labels.to(_CG.DEVICE)
                
                y_pred = self.model(images)
                
                # max returns (value, index)
                top_pred_val, top_pred_idx = torch.max(y_pred.data, 1)
                n_correct = top_pred_idx.eq(labels.view_as(top_pred_idx)).sum()
                
                # accuracy
                tot_samples += labels.size(0)
                tot_correct += n_correct
                cls_acc_dict = self.class_accuracies(cls_acc_dict, labels.view_as(top_pred_idx), top_pred_idx)

                # wandb
                all_labels = torch.cat((all_labels, labels), dim=0)
                all_cls_idxs = torch.cat((all_cls_idxs, top_pred_idx), dim=0)
                all_pred_vals = torch.cat((all_pred_vals, y_pred), dim=0)    
    
            # wandb
            wandb.log({
                    "pr_curve": wandb.plot.pr_curve(
                        all_labels.cpu().detach().numpy(),
                        all_pred_vals.cpu().detach().numpy(),
                        labels=list(self.dataset_wrapper.label_to_idx.keys())
                        )
                })
            wandb.log({
                    "confusion": wandb.plot.confusion_matrix(
                        y_true=all_labels.cpu().detach().numpy(),
                        preds=all_cls_idxs.cpu().detach().numpy(),
                        class_names=list(self.dataset_wrapper.label_to_idx.keys())
                        )
            })

            acc = tot_correct / tot_samples
            class_accuracy = { self.dataset_wrapper.idx_to_label[c.item()]: cls_acc_dict[c.item()][0] / cls_acc_dict[c.item()][1] for c in classes }

            Logger.instance().debug(f"Accuracy for each class: {class_accuracy}")
            Logger.instance().debug(f"Test accuracy on {len(self.test_info.subset.indices)} images: {acc:.3f}")
