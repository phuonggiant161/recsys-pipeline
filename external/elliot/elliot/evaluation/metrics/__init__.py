"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
Each metric is encapsulated in a specific package.

See the implementation of Precision metric for creating new per-user metrics.
See the implementation of Item Coverage for creating new cross-user metrics.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.accuracy.ndcg import nDCG, nDCGRendle2020
from elliot.evaluation.metrics.accuracy.precision import Precision
from elliot.evaluation.metrics.accuracy.recall import Recall
from elliot.evaluation.metrics.accuracy.hit_rate import HR
from elliot.evaluation.metrics.accuracy.mrr import MRR
from elliot.evaluation.metrics.accuracy.map import MAP
from elliot.evaluation.metrics.accuracy.mar import MAR
from elliot.evaluation.metrics.accuracy.f1 import F1, ExtendedF1
from elliot.evaluation.metrics.accuracy.DSC import DSC
from elliot.evaluation.metrics.accuracy.AUC import LAUC, AUC, GAUC

from elliot.evaluation.metrics.rating.mae import MAE
from elliot.evaluation.metrics.rating.mse import MSE
from elliot.evaluation.metrics.rating.rmse import RMSE

from elliot.evaluation.metrics.coverage import ItemCoverage, UserCoverage, NumRetrieved, UserCoverageAtN

from elliot.evaluation.metrics.diversity.gini_index import GiniIndex
from elliot.evaluation.metrics.diversity.shannon_entropy import ShannonEntropy
from elliot.evaluation.metrics.diversity.SRecall import SRecall

from elliot.evaluation.metrics.novelty.EFD import EFD, ExtendedEFD
from elliot.evaluation.metrics.novelty.EPC import EPC, ExtendedEPC

from elliot.evaluation.metrics.bias import ARP, APLT, ACLT, PopRSP, PopREO, ExtendedPopRSP, ExtendedPopREO

from elliot.evaluation.metrics.fairness.MAD import UserMADrating, ItemMADrating, UserMADranking, ItemMADranking
from elliot.evaluation.metrics.fairness.BiasDisparity import BiasDisparityBR, BiasDisparityBS, BiasDisparityBD
from elliot.evaluation.metrics.fairness.rsp import RSP
from elliot.evaluation.metrics.fairness.reo import REO

from elliot.evaluation.metrics.statistical_array_metric import StatisticalMetric
from elliot.evaluation.metrics.accuracy.user_group.user_group_metric import (
    make_tail_metric, make_u1_metric,
    make_pop_recall_metric, make_pop_ndcg_metric,
    make_user_group_metric,
)

_metric_dictionary = {
    "nDCG": nDCG,
    "nDCGRendle2020": nDCGRendle2020,
    "Precision": Precision,
    "Recall": Recall,
    "HR": HR,
    "MRR": MRR,
    "MAP": MAP,
    "MAR": MAR,
    "F1": F1,
    "ExtendedF1": ExtendedF1,
    "DSC": DSC,
    "LAUC": LAUC,
    "GAUC": GAUC,
    "AUC": AUC,
    "ItemCoverage": ItemCoverage,
    "UserCoverage": UserCoverage,
    "UserCoverageAtN": UserCoverageAtN,
    "NumRetrieved": NumRetrieved,
    "Gini": GiniIndex,
    "SEntropy": ShannonEntropy,
    "EFD": EFD,
    "ExtendedEFD": ExtendedEFD,
    "EPC": EPC,
    "ExtendedEPC": ExtendedEPC,
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "UserMADrating": UserMADrating,
    "ItemMADrating": ItemMADrating,
    "UserMADranking": UserMADranking,
    "ItemMADranking": ItemMADranking,
    "BiasDisparityBR": BiasDisparityBR,
    "BiasDisparityBS": BiasDisparityBS,
    "BiasDisparityBD": BiasDisparityBD,
    "SRecall": SRecall,
    "ARP": ARP,
    "APLT": APLT,
    "ACLT": ACLT,
    "PopRSP": PopRSP,
    "PopREO": PopREO,
    "ExtendedPopRSP": ExtendedPopRSP,
    "ExtendedPopREO": ExtendedPopREO,
    "RSP": RSP,
    "REO": REO,
    "Precision_u1": make_u1_metric(Precision),
    # User-activity group metrics (generalized U1: 5 bins by train interaction count)
    "Recall_u1":      make_user_group_metric(Recall, "u1"),
    "Recall_u2_5":    make_user_group_metric(Recall, "u2_5"),
    "Recall_u6_10":   make_user_group_metric(Recall, "u6_10"),
    "Recall_u11_20":  make_user_group_metric(Recall, "u11_20"),
    "Recall_u20plus": make_user_group_metric(Recall, "u20plus"),
    "nDCG_u1":        make_user_group_metric(nDCG, "u1"),
    "nDCG_u2_5":      make_user_group_metric(nDCG, "u2_5"),
    "nDCG_u6_10":     make_user_group_metric(nDCG, "u6_10"),
    "nDCG_u11_20":    make_user_group_metric(nDCG, "u11_20"),
    "nDCG_u20plus":   make_user_group_metric(nDCG, "u20plus"),
    "MAP_u1": make_u1_metric(MAP),
    "MRR_u1": make_u1_metric(MRR),
    "Recall_tail": make_tail_metric(Recall),
    "nDCG_tail": make_tail_metric(nDCG),
    # Item-popularity-group metrics (fixed bins, train-count based)
    "Recall_pop1":      make_pop_recall_metric("pop1"),
    "Recall_pop2_5":    make_pop_recall_metric("pop2_5"),
    "Recall_pop6_10":   make_pop_recall_metric("pop6_10"),
    "Recall_pop11_20":  make_pop_recall_metric("pop11_20"),
    "Recall_pop20plus": make_pop_recall_metric("pop20plus"),
    "nDCG_pop1":        make_pop_ndcg_metric("pop1"),
    "nDCG_pop2_5":      make_pop_ndcg_metric("pop2_5"),
    "nDCG_pop6_10":     make_pop_ndcg_metric("pop6_10"),
    "nDCG_pop11_20":    make_pop_ndcg_metric("pop11_20"),
    "nDCG_pop20plus":   make_pop_ndcg_metric("pop20plus"),
}

_lower_dict = {k.lower(): v for k, v in _metric_dictionary.items()}


def parse_metrics(metrics):
    return [_lower_dict[m.lower()] for m in metrics if m.lower() in _lower_dict.keys()]


def parse_metric(metric):
    metric = metric.lower()
    return _lower_dict[metric] if metric in _lower_dict.keys() else None
