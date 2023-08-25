import torch

from typing import Optional, Tuple
from torch.nn import functional as F
from torch.nn.modules import Module

from src.models.FSL.ProtoNet.proto_extra_modules import ProtoEnhancements
from lib.glass_defect_dataset.config.consts import General as _CG


class ProtoTools:

    ZERO = torch.Tensor([.0]).to(_CG.DEVICE)
    MARGIN = torch.Tensor([.2]).to(_CG.DEVICE)

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

        square_dist = torch.pow(x - y, 2).sum(2)
        
        if sqrt: return torch.sqrt(square_dist)
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

        # floating point arithmetic fix
        sim = torch.nn.CosineSimilarity(dim=-1)(x, y)
        dist = torch.ones_like(sim) - sim
        mask = torch.isclose(dist, ProtoTools.ZERO, rtol=0, atol=1e-6)
        dist[mask] = 0.0

        return dist

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

        if enhance.name == ProtoEnhancements.MOD_IPN:
            proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos, sqrt_eucl)
            mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
            alphas = enhance.module_list[0].forward(mean_dists)
            dists = alphas * dists
        elif enhance.name == ProtoEnhancements.MOD_DIST:
            proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos, sqrt_eucl)
            query_dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos, sqrt_eucl)
            mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
            dists = ProtoTools.euclidean_dist(query_dists, mean_dists, sqrt_eucl)
        else:
            pass
        
        return dists
    

class ProtoLoss:

    def __init__(self, enhance: ProtoEnhancements, sqrt_eucl: bool):
        self.enhance = enhance
        self.sqrt_eucl = sqrt_eucl

        self.acc: torch.Tensor = torch.FloatTensor([]).to(_CG.DEVICE)
        self.proto_loss: torch.Tensor = torch.FloatTensor([]).to(_CG.DEVICE)
        self.contrastive_loss: Optional[torch.Tensor] = None

        # must be updated
        self.loss: torch.Tensor = self.__init_loss()
        self.loss_dict: dict = self.__init_loss_dict()

    def compute_loss(self, recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int):
        s_batch, q_batch = ProtoTools.split_support_query(recons, target, n_way, n_support, n_query)
        
        if self.enhance.name == ProtoEnhancements.MOD_APN:
            self.contrastive_loss = self._contrastive_loss(s_batch.view(n_way * n_support, -1), n_way, n_support)

        loss, self.acc = self._proto_loss(s_batch, q_batch, n_way, n_query)
        
        # udpate value
        self.loss = loss
        self.loss_dict["proto_loss"] = loss
        if self.contrastive_loss is not None:
            self.loss = loss + self.contrastive_loss
            self.loss_dict["contrastive_loss"] = self.contrastive_loss
        
        # final
        self.loss_dict["total_loss"] = self.loss

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

    def _contrastive_loss(self, xs_emb: torch.Tensor, n_classes: int, n_support: int) -> torch.Tensor:
        # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/14#issuecomment-952603383
        #xs_emb = xs_emb / torch.norm(xs_emb, dim=-1, keepdim=True)

        dists_full = ProtoTools.euclidean_dist(xs_emb, xs_emb, sqrt=self.sqrt_eucl) # HERE
        cosine_dists = ProtoTools.cosine_distance(xs_emb, xs_emb)
        
        dists = cosine_dists.triu(diagonal=1) # distance is symmetric, remove them to not compute twice
        #dists = dists_full.triu(diagonal=1)

        if dists.size(0) != dists.size(1) != n_classes * n_support:
            raise ValueError(f"Tensor must be square-shaped. Found t = {dists.shape}, N = {n_classes} * K = {n_support}")

        r = torch.arange(n_classes * n_support).to(_CG.DEVICE)

        ## numerator
        numerator_select = r.view(-1, n_support).unsqueeze(1).expand(n_classes, n_support, -1)
        numerator_sym = dists.view(n_classes, n_support, -1).gather(2, numerator_select)
        numerator_mask = torch.triu(torch.ones_like(numerator_select), diagonal=1).bool()
        numerator = numerator_sym[numerator_mask]

        ## denominator
        all_idxs = r.view(-1, n_classes * n_support).unsqueeze(1).expand(n_classes, n_support, n_classes * n_support)
        mask = ~torch.any(numerator_select.view(n_classes, n_support, n_support, 1) == all_idxs.view(n_classes, n_support, 1, n_classes * n_support), dim=2)
        mask_sym = torch.triu(torch.ones_like(dists), diagonal=1).bool().view(n_classes, n_support, -1)
        final_mask = mask & mask_sym
        denominator = dists.view(n_classes, n_support, -1)[final_mask]

        # we want a balance for the number of positive and negative pairs, so sample n_support comparison
        rand_start = torch.randint(0, denominator.size(0) - numerator.size(0), (1,)).item()
        sample_den = denominator[rand_start : rand_start + numerator.size(0)]
        
        positive_pairs = torch.pow(numerator.sum(dim=-1), 2)
        negative_pairs = sample_den.sum(dim=-1)
        neg_margin = torch.pow(torch.max(ProtoTools.ZERO, ProtoTools.MARGIN - negative_pairs), 2)

        loss = torch.div((positive_pairs + neg_margin), 2 * numerator.size(0))

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
        acc_vals = { c: y_hat[c].eq(target_inds.squeeze(2)[c]).float().mean() for c in range(n_way) }

        self.acc_overall = torch.cat((self.acc_overall, acc_overall.flatten()))
        self.y_hat = torch.cat((self.y_hat, y_hat.flatten()))
        self.target_inds = torch.cat((self.target_inds, target_inds.flatten()))

        return acc_overall, { v: acc_vals[i] for i, v in enumerate(mapping.values()) }
