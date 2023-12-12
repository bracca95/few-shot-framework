import os
import json
import torch
import pandas as pd

from typing import Optional, Tuple, List
from torch.nn import functional as F

from src.models.FSL.ProtoNet.proto_extra_modules import ProtoEnhancements
from lib.glass_defect_dataset.config.consts import General as _CG
from lib.glass_defect_dataset.src.utils.tools import Tools, Logger


class ProtoTools:

    ZERO = torch.tensor(0.0, dtype=torch.float, device=_CG.DEVICE, requires_grad=False)

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor, sqrt: bool=True) -> torch.Tensor:
        """Compute euclidean distance between two tensors

        The size has a unique constraint the fact that `d` (feature vector length must be the same). This function can
        be applied to compare queries to class prototypes, or to compare distances between query pairs. In the latter
        case, N = M
        
        Args:
            x (torch.Tensor): size N x D
            y (torch.Tensor): size M x D
            sqrt (bool): use square rooted distance 

        Returns:
            torch.Tensor

        Raises:
            ValueError if `d` for tensor `x` differs from `d` of tensor `y`
        """
        
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise ValueError(f"Tensor shapes must be the same in -1: x: {x.shape}, y: {y.shape}")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        square_dist = torch.pow(x - y + 1e-6, 2).sum(2)
        
        if sqrt:
            return torch.sqrt(square_dist) # same as nn.F.pairwise_distance(x, y)
        
        return square_dist
    
    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise ValueError(f"Tensor shapes must be the same in -1: x: {x.shape}, y: {y.shape}")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        sim = torch.nn.CosineSimilarity(dim=-1)(x, y)
        return 1 - sim

    @staticmethod
    def split_batch(img: torch.Tensor, labels: torch.Tensor, n_way: int, n_support: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # the sampler is supposed to ensure class correctness (i.e. number of unique classes == n_way)
        classes = torch.unique(labels)

        # supposing n_way is shared for support and query (may not be the case)
        n_query = labels.size(0) // n_way - n_support
        class_idx = torch.stack(list(map(lambda x: torch.where(labels == x)[0], classes)))
        support_idxs, query_idxs = torch.split(class_idx, [n_support, n_query], dim=1)

        support_set = img[support_idxs.flatten()]
        query_set = img[query_idxs.flatten()]

        # support_batch, support_label, query_batch, query_label
        return support_set, support_idxs.flatten(), query_set, query_idxs.flatten()
    
    @staticmethod
    def split_support_query(recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # check correct input
        classes = torch.unique(target)
        if not n_way == len(classes):
            raise ValueError(f"number of unique classes ({len(classes)}) must match config n_way ({n_way})")
        
        if not target.size(0) // len(classes) == n_support + n_query:
            raise ValueError(f"({target.size(0) // len(classes)}) != support ({n_support}) + query ({n_query})")
        
        class_idx = torch.stack(list(map(lambda x: torch.where(target == x)[0], classes)))  # shape = (n_way, s+q)
        support_idxs, query_idxs = torch.split(class_idx, [n_support, n_query], dim=1)

        support_set = recons[support_idxs.flatten()].view(n_way, n_support, -1)
        query_set = recons[query_idxs.flatten()].view(n_way, n_query, -1)

        return support_set, query_set
    
    @staticmethod
    def get_dists(s_batch: torch.Tensor, q_batch: torch.Tensor, enhance: ProtoEnhancements, **kwargs) -> torch.Tensor:
        n_classes, n_query, n_feat = (q_batch.shape)

        sqrt_eucl = False
        if "sqrt_eucl" in kwargs:
            sqrt_eucl: bool = kwargs["sqrt_eucl"]

        ### vanilla protonet
        protos = torch.mean(s_batch, dim=1)
        dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos, sqrt_eucl)
        ### EOF: vanilla protonet

        if enhance.name == ProtoEnhancements.ENH_IPN:
            proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos, sqrt_eucl)
            mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
            alphas = enhance.extra_modules[ProtoEnhancements.MOD_DISTSCALE].forward(mean_dists)
            dists = dists / alphas
        elif enhance.name == ProtoEnhancements.ENH_DIST:
            proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos, sqrt_eucl)
            query_dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos, sqrt_eucl)
            mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
            dists = ProtoTools.euclidean_dist(query_dists, mean_dists, sqrt_eucl)
        else:
            pass
        
        return dists
    

class ProtoLoss:

    GAMMA = 0.5
    MARGIN = torch.tensor(0.2, dtype=torch.float, device=_CG.DEVICE, requires_grad=False)

    def __init__(self, enhance: ProtoEnhancements, sqrt_eucl: bool, tot_epochs: int, weighted: bool):
        self.enhance = enhance
        self.sqrt_eucl = sqrt_eucl
        self.weighted = weighted

        # not used in test, but making it Optional would complicate things
        first_half = torch.linspace(1e-6, ProtoLoss.GAMMA, tot_epochs // 2)
        second_half = torch.full((tot_epochs - len(first_half),), ProtoLoss.GAMMA)
        self.gammas = torch.cat((first_half, second_half))
        
        self.acc: torch.Tensor = torch.FloatTensor([]).to(_CG.DEVICE)
        self.proto_loss: torch.Tensor = torch.FloatTensor([]).to(_CG.DEVICE)
        self.contrastive_loss: Optional[torch.Tensor] = None
        self.soft_loss: Optional[torch.Tensor] = None

        # must be updated
        self.loss: torch.Tensor = self.__init_loss()
        self.loss_dict: dict = self.__init_loss_dict()

    def compute_loss(self, recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int, epoch: int):
        s_batch, q_batch = ProtoTools.split_support_query(recons, target, n_way, n_support, n_query)

        ## compute required losses 
        # contrastive loss
        if self.enhance.name == ProtoEnhancements.ENH_APN or self.enhance.name == ProtoEnhancements.ENH_CONTR_LSTM:
            self.contrastive_loss = self._soft_nn_loss(s_batch.view(n_way * n_support, -1), n_way, n_support)
            self.loss_dict["contrastive_loss"] = self.contrastive_loss.detach()

        if self.enhance.name == ProtoEnhancements.ENH_AUTOCORR:
            self.soft_loss = self._soft_nn_loss(s_batch.view(n_way * n_support, -1), n_way, n_support)
            self.contrastive_loss = self._barlow_cc_loss(s_batch.view(n_way * n_support, -1), n_way, n_support)
            self.loss_dict["soft_loss"] = self.soft_loss.detach()
            self.loss_dict["contrastive_loss"] = self.contrastive_loss.detach()

        # prototypical loss
        self.proto_loss, self.acc = self._proto_loss(s_batch, q_batch, n_way, n_query)
        self.loss_dict["proto_loss"] = self.proto_loss.detach()
        ##

        # udpate values
        self.loss = self.proto_loss
        
        if self.contrastive_loss is not None:
            if self.weighted:
                gamma = self.gammas[epoch]
                self.loss = ((1.0 - gamma) * self.proto_loss) + (gamma * self.contrastive_loss)
            else:
                self.loss = self.loss + self.contrastive_loss
        
        if self.soft_loss is not None:
            self.loss = self.loss + self.soft_loss
        
        # final
        self.loss_dict["total_loss"] = self.loss.detach()

    def _proto_loss(self, s_batch: torch.Tensor, q_batch: torch.Tensor, n_way: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = ProtoTools.get_dists(s_batch, q_batch, self.enhance, sqrt_eucl=self.sqrt_eucl)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
        target_inds = torch.arange(0, n_way).to(_CG.DEVICE)
        target_inds = target_inds.view(n_way, 1, 1)
        target_inds = target_inds.expand(n_way, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val, acc_val
    
    def _soft_nn_loss(self, xs_emb: torch.Tensor, n_classes: int, n_support: int) -> torch.Tensor:
        bs = n_classes * n_support
        main_mask = ~torch.eye(bs, dtype=torch.bool, device=_CG.DEVICE).view(bs, 1, bs)
        r = torch.arange(bs).to(_CG.DEVICE).detach()

        dists = ProtoTools.euclidean_dist(xs_emb, xs_emb).unsqueeze(0).view(n_classes, n_support, -1)

        # numerator
        num_sel = r.view(-1, n_support).unsqueeze(1).expand(n_classes, n_support, -1)
        numerator_mask = torch.zeros_like(dists, dtype=torch.bool, requires_grad=False)
        numerator_mask = numerator_mask.scatter_(2, num_sel, True).view(bs, 1, bs)
        numerator_mask = numerator_mask & main_mask
        numerator = dists.view(bs, 1, -1)[numerator_mask].view(bs, 1, -1)
        exp_num_sum = torch.exp(-numerator).sum(dim=-1)

        # denominator
        denominator = dists.view(bs, 1, -1)[main_mask].view(bs, 1, -1)
        exp_den_sum = torch.exp(-denominator).sum(dim=-1)

        # log sum
        log_sum = torch.log(torch.div(exp_num_sum, (exp_den_sum + 1e-6)) + 1e-6).flatten().sum()
        
        # loss
        loss = torch.div(-log_sum, bs)
        return loss

    def _barlow_cc_loss(self, xs_emb: torch.Tensor, n_classes: int, n_support: int) -> torch.Tensor:
        const = 5e-3
        xs_norm = F.normalize(xs_emb, p=2, dim=-1) # no need to norm since need not be orthonormal
        autocorr = torch.matmul(xs_norm, xs_norm.t())
        id_mat = torch.eye(autocorr.size(0), device=autocorr.device)

        # loss
        c_diff = (autocorr - id_mat).pow(2)
        mask_off_diag = ~torch.eye(c_diff.size(0), dtype=torch.bool)
        c_diff_off_diag = c_diff[mask_off_diag] * const
        loss = c_diff_off_diag.sum() #+ c_diff.diag().sum()

        return loss

    def _triplet_loss(self, xs_emb: torch.Tensor, n_classes: int, n_support: int) -> torch.Tensor:
        """
        
        The 'for loop' way:
        result_neg_list = []
        result_pos_list = []
        for n in range(n_classes):
            lower_bound = n * n_support
            upper_bound = (n + 1) * n_support - 1
            
            neg = (sorted_indices[n, :, :n_support] < lower_bound) | (sorted_indices[n, :, :n_support] > upper_bound)
            pos = (sorted_indices[n, :, n_support:] >= lower_bound) & (sorted_indices[n, :, n_support:] <= upper_bound)
            
            result_neg = sorted_dists[n, :, :n_support][neg]
            result_pos = sorted_dists[n, :, n_support:][pos]

            result_neg_list.append(result_neg)
            result_pos_list.append(result_pos)

        # Convert the lists to tensors
        result_neg_tensor = torch.cat(result_neg_list)
        result_pos_tensor = torch.cat(result_pos_list)
        """

        dists = ProtoTools.cosine_distance(xs_emb, xs_emb)
        sorted_dists, sorted_indices = torch.sort(dists.view(n_classes, n_support, -1))

        lb = torch.arange(start=0, end=(n_support * n_classes), step=n_support, device=_CG.DEVICE).view(n_classes, 1, 1)
        ub = lb + n_support - 1

        # negatives are != classes found as same (close), positives are == class, found as other (apart)
        neg = (sorted_indices[:, :, :n_support] < lb) | (sorted_indices[:, :, :n_support] > ub)
        pos = (sorted_indices[:, :, n_support:] >= lb) & (sorted_indices[:, :, n_support:] <= ub)

        result_neg = sorted_dists[:, :, :n_support][neg]
        result_pos = sorted_dists[:, :, n_support:][pos]

        loss = torch.max((result_pos - result_neg + ProtoLoss.MARGIN), ProtoTools.ZERO).mean()
        
        return loss
    
    def _lifted_structured_loss(self, xs_emb: torch.Tensor, n_classes: int, n_support: int) -> torch.Tensor:
        # norm_emb = xs_emb / torch.linalg.norm(xs_emb, dim=-1, ord=2, keepdim=True)                    # L2 row-vector norm
        # norm_emb = (xs_emb - xs_emb.mean()) / (xs_emb.std() + 1e-6)                                   # z-score normalization aka standardization (batch-norm)
        # norm_emb = xs_emb / torch.linalg.matrix_norm(xs_emb, ord='fro')                               # matrix norm
        # norm_emb = torch.div(xs_emb - torch.min(xs_emb), torch.max(xs_emb) - torch.min(xs_emb))       # classic normalization formula
        
        dists = ProtoTools.euclidean_dist(xs_emb, xs_emb, sqrt=self.sqrt_eucl)
        
        if dists.size(0) != dists.size(1) or dists.size(1) != n_classes * n_support:
            raise ValueError(f"Tensor must be square-shaped. Found t = {dists.shape}, N = {n_classes} * K = {n_support}")

        r = torch.arange(n_classes * n_support).to(_CG.DEVICE)

        ## numerator
        numerator_select = r.view(-1, n_support).unsqueeze(1).expand(n_classes, n_support, -1)
        numerator = dists.view(n_classes, n_support, -1).gather(2, numerator_select)

        ## denominator: normalize out of the same class
        all_idxs = r.view(-1, n_classes * n_support).unsqueeze(1).expand(n_classes, n_support, n_classes * n_support)
        mask = ~torch.any(numerator_select.view(n_classes, n_support, n_support, 1) == all_idxs.view(n_classes, n_support, 1, n_classes * n_support), dim=2)
        den_select = torch.masked_select(all_idxs, mask).view(n_classes, n_support, -1)
        den_elems = dists.view(n_classes, n_support, -1).gather(2, den_select)

        """brutal way
            result = torch.empty_like(numerator)
            for c in range(numerator.size(0)):
                for i in range(numerator.size(1)):
                    for j in range(numerator.size(1)):
                        if i == j: continue
                        neg = torch.log(torch.sum(torch.exp((ProtoLoss.MARGIN - den_elems[c][i]))) + torch.sum(torch.exp((ProtoLoss.MARGIN - den_elems[c][j]))))
                        result[c][i][j] = numerator[c][i][j] + neg
        """
        
        exp_neg = torch.exp(ProtoLoss.MARGIN - den_elems)
        sum_exp_neg = exp_neg.sum(dim=2)
        neg_broadcasted = sum_exp_neg.unsqueeze(1) + sum_exp_neg.unsqueeze(2)
        result = numerator + torch.log(neg_broadcasted)
        
        # remove self comparison
        result = result * (1+1e-6 - torch.eye(result.size(1)).to(_CG.DEVICE))
        
        loss = torch.div(torch.sum(torch.pow(torch.max(ProtoTools.ZERO, result), 2)), 2 * torch.numel(numerator))
        return loss
    
    def __init_loss(self):
        return torch.FloatTensor([]).to(_CG.DEVICE)

    def __init_loss_dict(self):
        # mandatory
        loss_dict = { "proto_loss": self.proto_loss}
        
        # optional entries
        if self.contrastive_loss is not None:
            loss_dict["contrastive_loss"] = self.contrastive_loss

        return loss_dict
    
    @staticmethod
    def append_epoch_loss(train_loss: dict, current_loss: dict):
        if len(train_loss) == 0:
            train_loss.update({ k: [] for k in current_loss.keys() })

        [ train_loss[k].append(v.item()) for k, v in current_loss.items() ]


class TestResult:
    def __init__(self):
        self.acc_overall = torch.Tensor().to(_CG.DEVICE)
        self.y_hat = torch.Tensor().to(_CG.DEVICE)
        self.target_inds = torch.Tensor().to(_CG.DEVICE)

    def proto_test(self, recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int, enhance: ProtoEnhancements, sqrt_eucl: bool):
        mapping = { i: i for i in range(n_way) }
        
        s_batch, q_batch = ProtoTools.split_support_query(recons, target, n_way, n_support, n_query)
        dists = ProtoTools.get_dists(s_batch, q_batch, enhance, sqrt_eucl=sqrt_eucl)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
        target_inds = torch.arange(0, n_way).to(_CG.DEVICE)
        target_inds = target_inds.view(n_way, 1, 1)
        target_inds = target_inds.expand(n_way, n_query, 1).long()

        _, y_hat = log_p_y.max(2)

        acc_overall = y_hat.eq(target_inds.squeeze(2)).float().mean()
        recall = { c: y_hat[c].eq(target_inds.squeeze(2)[c]).float().mean() for c in range(n_way) }

        self.acc_overall = torch.cat((self.acc_overall, acc_overall.flatten()))
        self.y_hat = torch.cat((self.y_hat, y_hat.flatten()))
        self.target_inds = torch.cat((self.target_inds, target_inds.flatten()))

        return acc_overall, { v: recall[i] for i, v in enumerate(mapping.values()) }
    

class InferenceResult:

    COL_PATH = "path"
    COL_CLASS = "class"
    COL_PREDICT = "predict"

    def __init__(
        self,
        enhance: ProtoEnhancements,
        s_emb: torch.Tensor,
        n_way: int,
        k_shot_s: int,
        dim: int,
        idx_to_label: dict,
        sqrt_eucl: bool
    ):
        import pandas as pd

        self.model = enhance
        self.s_emb = s_emb
        self.n_way = n_way
        self.k_shot_s = k_shot_s
        self.dim = dim
        self.idx_to_label = idx_to_label
        self.sqrt_eucl = sqrt_eucl

        # overall values
        self.path: List[str] = list()
        self.y = torch.tensor([], dtype=torch.float, device=_CG.DEVICE, requires_grad=False)
        self.y_hat = torch.tensor([], dtype=torch.float, device=_CG.DEVICE, requires_grad=False)
        self.table = pd.DataFrame(columns=[self.COL_PATH, self.COL_CLASS, self.COL_PREDICT])

    def proto_inference(self, query_loader):
        # load all queries in batches
        for qx, qy, path in query_loader:
            qx, qy = qx.to(_CG.DEVICE), qy.to(_CG.DEVICE)
            q_emb = self.model.base_model(qx)

            # manage last batch (likely to have less samples than n_way * k_shot and we do not want padding)
            s_emb = self.s_emb.view(self.n_way, self.k_shot_s, -1)
            if q_emb.size(0) % self.n_way == 0:
                q_emb = q_emb.view(self.n_way, -1, self.dim)
            else:
                q_emb = q_emb.view(-1, q_emb.size(0), self.dim)

            # predict
            dists = ProtoTools.get_dists(s_emb, q_emb, self.model, sqrt_eucl=True)
            log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
            _, y_hat = log_p_y.max(-1)
            
            # collect values
            self.path.extend(path)
            self.y = torch.cat((self.y, qy))
            self.y_hat = torch.cat((self.y_hat, y_hat))

    def compute_accuracy(self):
        # accuracy, precision and recall
        acc_overall = self.y_hat.eq(self.y).float().mean()
        acc_vals = { 
            self.idx_to_label[c]: torch.div(
                (self.y_hat.eq(c) & self.y.eq(c)).sum() + (self.y_hat.ne(c) & self.y.ne(c)).sum(),
                (self.y_hat.eq(c) & self.y.eq(c)).sum() + (self.y_hat.ne(c) & self.y.ne(c)).sum() + \
                        (self.y_hat.eq(c) & self.y.ne(c)).sum() + (self.y_hat.ne(c) & self.y.eq(c)).sum()
            ).item()
            for c in torch.unique(self.y).tolist()
        }
        recall = {
            self.idx_to_label[c]: torch.div(
                    (self.y_hat.eq(c) & self.y.eq(c)).sum(),
                    (self.y_hat.eq(c) & self.y.eq(c)).sum() + (self.y_hat.ne(c) & self.y.eq(c)).sum()
                ).item()
            for c in torch.unique(self.y).tolist()
        }
        precision = {
            self.idx_to_label[c]: torch.div(
                (self.y_hat.eq(c) & self.y.eq(c)).sum(),
                (self.y_hat.eq(c) & self.y.eq(c)).sum() + (self.y_hat.eq(c) & self.y.ne(c)).sum()
            ).item()
            for c in torch.unique(self.y).tolist()
        }

        self.table[self.COL_PATH] = self.path
        self.table[self.COL_CLASS] = [self.idx_to_label[l] for l in self.y.cpu().numpy().astype(int).tolist()]
        self.table[self.COL_PREDICT] = [self.idx_to_label[l] for l in self.y_hat.cpu().numpy().astype(int).tolist()]
        self.table.to_csv(f"output/score.csv", sep=",")

        import os
        import wandb
        from PIL import Image

        mismatch_dir = os.path.join(os.getcwd(), "output", "mismatch")
        if not os.path.exists(mismatch_dir):
            os.makedirs(mismatch_dir)
        
        for _, row in self.table.iterrows():
            if row[self.COL_CLASS] == row[self.COL_PREDICT]:
                continue

            # only save if mismatch
            image = Image.open(row[self.COL_PATH]).convert("L")
            name, ext = row[self.COL_PATH].rsplit(".", 1)
            image.save(os.path.join(mismatch_dir, f"{os.path.basename(name)}_{row[self.COL_PREDICT]}.{ext}"))
        
        wandb.save(f"{os.path.join(mismatch_dir, '*.png')}", base_path=os.getcwd())
        wandb.save(f"{os.path.join(os.getcwd(), 'output/score.csv')}", base_path=os.getcwd())
        return acc_overall, acc_vals, recall, precision
    

class FullInferenceResult:

    def __init__(
        self,
        enhance: ProtoEnhancements,
        s_emb: torch.Tensor,
        n_way: int,
        k_shot_s: int,
        dim: int,
        idx_to_label_support: dict,
        idx_to_label_query: dict,
        bb_file: pd.DataFrame,
        sqrt_eucl: bool
    ):
        self.model = enhance
        self.s_emb = s_emb
        self.n_way = n_way
        self.k_shot_s = k_shot_s
        self.dim = dim
        self.idx_to_label_support = idx_to_label_support
        self.idx_to_label_query = idx_to_label_query
        self.bb_file  = bb_file
        self.sqrt_eucl = sqrt_eucl

        # overall values
        self.path: List[str] = list()
        self.qy = torch.tensor([], dtype=torch.float, device=_CG.DEVICE, requires_grad=False)
        self.y_hat = torch.tensor([], dtype=torch.float, device=_CG.DEVICE, requires_grad=False)

    def proto_inference(self, query_loader):
        # load all queries in batches
        for qx, qy, path in query_loader:
            qx, qy = qx.to(_CG.DEVICE), qy.to(_CG.DEVICE)
            q_emb = self.model.base_model(qx)

            # manage last batch (likely to have less samples than n_way * k_shot and we do not want padding)
            s_emb = self.s_emb.view(self.n_way, self.k_shot_s, -1)
            if q_emb.size(0) % self.n_way == 0:
                q_emb = q_emb.view(self.n_way, -1, self.dim)
            else:
                q_emb = q_emb.view(-1, q_emb.size(0), self.dim)

            # predict
            dists = ProtoTools.get_dists(s_emb, q_emb, self.model, sqrt_eucl=True)
            log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
            _, y_hat = log_p_y.max(-1)

            # collect values
            self.path.extend(path)
            self.qy = torch.cat((self.qy, qy))
            self.y_hat = torch.cat((self.y_hat, y_hat))

    def provide_output(self):
        df = self.bb_file.copy()
        prediction = { }

        new_class_list: List[str] = list()
        defect_id_list: List[int] = list()

        y_hat = [self.idx_to_label_support[l] for l in self.y_hat.cpu().numpy().astype(int).tolist()]
        
        for i in range(len(self.path)):
            _, def_idplus = os.path.basename(self.path[i]).rsplit("_did_", 1)
            defect_id = int(def_idplus.rsplit("_vid_", 1)[0])
            new_class = y_hat[i]

            new_class_list.append(new_class)
            defect_id_list.append(defect_id)

        new_df = pd.DataFrame({"#id_defect": defect_id_list, "#class_key": new_class_list})
        filtered_df = df[df["#id_defect"].isin(defect_id_list)]

        # merge the dataframes based on the id_defect column
        merged_df = pd.merge(filtered_df, new_df, on="#id_defect", how="left", suffixes=("_original", "_new"))
        
        # stat matrix (somehow)
        for idx, row in merged_df.iterrows():
            if f'{row["#class_key_original"]}_as_{row["#class_key_new"]}' in prediction.keys():
                prediction[f'{row["#class_key_original"]}_as_{row["#class_key_new"]}'] += 1
            else:
                prediction[f'{row["#class_key_original"]}_as_{row["#class_key_new"]}'] = 1
            
            # store total number
            if f'TOT_{row["#class_key_original"]}' in prediction.keys():
                prediction[f'TOT_{row["#class_key_original"]}'] += 1
            else:
                prediction[f'TOT_{row["#class_key_original"]}'] = 1
        
        # replace names
        merged_df["#class_key_original"] = merged_df["#class_key_new"].fillna(merged_df["#class_key_original"])
        merged_df = merged_df.rename(columns={"#class_key_original": "#class_key"})
        merged_df = merged_df.drop(["#class_key_new"], axis=1)

        # save
        merged_df.to_csv(os.path.join(os.getcwd(), "output", "bounding_boxes.csv"))
        with open(os.path.join(os.getcwd(), "output", "out_stat_matrix.json"), "w") as f:
            json.dump(prediction, f, indent=4)
        