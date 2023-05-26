import os
import sys
import torch

from typing import Union, List
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from src.datasets.defectviews import DefectViews
from config.consts import General as _CG
from config.consts import SubsetsDict
from src.train_test.routine import TrainTest
from src.utils.config_parser import Config
from src.utils.tools import Logger, TBWriter, Tools


class StandardRoutine(TrainTest):

    def __init__(self, model: nn.Module, dataset: DefectViews, subsets_dict: SubsetsDict):
        super().__init__(model, dataset, subsets_dict)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(_CG.DEVICE)

        # tensorboard
        self.writer = TBWriter.instance().get_writer()

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
        top_pred = y_pred.argmax(1, keepdim=True)           # select the max class (the one with the highest score)
        correct = top_pred.eq(y.view_as(top_pred)).sum()    # count the number of correct predictions
        return correct.float() / y.shape[0]                 # compute percentage of correct predictions (accuracy score)
    
    def init_loader(self, config: Config, split_set: str):
        current_subset = self.get_subset_info(split_set)
        if current_subset.subset is None:
            return None
        
        return DataLoader(current_subset.subset, batch_size=config.batch_size)

    def train(self, config: Config):
        Logger.instance().debug("Start training")
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        trainloader = self.init_loader(config, self.train_str)
        valloader = self.init_loader(config, self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        
        # tensorboard
        example_data, examples_target = next(iter(trainloader))
        shape = config.crop_size if config.image_size is None else config.image_size
        #self.writer.add_graph(self.model, example_data.to(self.device))
        img_grid = make_grid(example_data)
        self.writer.add_image('images', img_grid)
        self.writer.close()

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

        for eidx, epoch in enumerate(range(config.epochs)):
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

            if epoch_acc > best_acc and epoch > 0:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), best_model_path)
                Logger.instance().debug(f"saving model at iteration {epoch}.")

            # tensorboard
            loss_dict = { "epoch_loss": epoch_loss }
            acc_dict = { "epoch_acc": epoch_acc }

            ## VALIDATION
            if valloader is not None:
                epoch_loss_eval, epoch_acc_eval = self.validate(config, valloader)
                if epoch_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), val_model_path)

                # tensorboard
                loss_dict["epoch_loss_eval"] = epoch_loss_eval
                acc_dict["epoch_acc_eval"] = epoch_acc_eval    
            ## EOF: VALIDATION

            # tensorboard
            # https://stackoverflow.com/questions/48951136/plot-multiple-graphs-in-one-plot-using-tensorboard
            self.writer.add_scalars("Loss", loss_dict, epoch)
            self.writer.add_scalars("Accuracy", acc_dict, epoch)

            # save last model
            if eidx == config.epochs-1:
                pth_path = last_val_model_path if valloader is not None else last_model_path
                Logger.instance().debug(f"saving last epoch model named `{os.path.basename(pth_path)}`")
                torch.save(self.model.state_dict(), pth_path)

        self.writer.close()

    def validate(self, config: Config, valloader: DataLoader):
        Logger.instance().debug("Validating!")
        self.model.eval()
        
        tot_samples = 0
        tot_correct = 0

        loss = 0
        
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

        acc = tot_correct / tot_samples

        Logger.instance().debug(f"Avg Val Loss: {loss}, Avg Val Acc: {acc}")

        return loss, acc
    
    def test(self, config: Config, model_path: str):
        Logger.instance().debug("Start testing")
        
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")
        
        try:
            model_path = Tools.validate_path(model_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)

        self.model.load_state_dict(torch.load(model_path))
        testloader = self.init_loader(config, self.test_str)

        prcurve_labels = []
        prcurve_predic = []
        
        self.model.eval()
        with torch.no_grad():
            tot_samples = 0
            tot_correct = 0
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

                # precision-recall curve
                prcurve_labels.extend(labels)
                prcurve_predic.append(y_pred)

            acc = tot_correct / tot_samples
            Logger.instance().debug(f"Test accuracy on {len(self.test_info.subset.indices)} images: {acc:.3f}")

        # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve
        # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#assessing-trained-models-with-tensorboard
        test_probs = torch.cat([torch.stack(tuple(batch)) for batch in prcurve_predic])
        for i in range(len(self.test_info.info_dict)):
            truth = list(map(lambda x: x == i, prcurve_labels))
            probs = test_probs[:, i]
            self.writer.add_pr_curve(str(i), torch.Tensor(truth), probs.cpu(), global_step=0, num_thresholds=1000)
            self.writer.close()