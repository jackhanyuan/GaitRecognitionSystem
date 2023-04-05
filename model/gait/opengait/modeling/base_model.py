"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_test, etc.
 The api of the base model is run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast

from . import backbones
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from tools import Odict, mkdir
from tools import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from tools import NoOp
from tools import get_msg_mgr



class BaseModel(nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training=False):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        self.build_network(cfgs['model_cfg'])

        self.msg_mgr.log_info(cfgs['data_cfg'])

        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

        self.train(training)

        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def get_loader(self, data_cfg, train=False):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = torch.load(save_name, map_location=torch.device(self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs

        trf_cfgs = self.engine_cfg['transform']
        seq_trfs = get_transform(trf_cfgs)
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()

        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs

        return ipts, labs, typs, vies, seqL

    def inference(self, rank=0):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:

            ipts = self.inputs_pretreament(inputs)

            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""

        with torch.no_grad():
            info_dict = model.inference()

        loader = model.test_loader
        label_list = loader.dataset.label_list
        types_list = loader.dataset.types_list
        views_list = loader.dataset.views_list

        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
            eval_func = model.cfgs['evaluator_cfg']["eval_func"]
        else:
            eval_func = 'identification'
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(
            eval_func, model.cfgs["evaluator_cfg"], ['metric'])
        try:
            dataset_name = model.cfgs['data_cfg']['test_dataset_name']
        except:
            dataset_name = model.cfgs['data_cfg']['dataset_name']
        return eval_func(info_dict, dataset_name, **valid_args)
