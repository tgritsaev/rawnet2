# original code: https://github.com/XuMuK1/dla2023/blob/2023/hw5_as/calculate_eer.py
import torch


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = torch.concatenate((target_scores, nontarget_scores))
    labels = torch.concatenate((torch.ones(target_scores.size), torch.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = torch.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = torch.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (torch.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = torch.concatenate((torch.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = torch.concatenate((torch.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = torch.concatenate((torch.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    """
    Returns equal error rate (EER) and the corresponding threshold.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = torch.abs(frr - far)
    min_index = torch.argmin(abs_diffs)
    eer = torch.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


class EqualErrorRate:
    def __init__(self):
        self.name = "equal_error_rate"

    def __call__(self, target, pred, **kwargs):
        # target = target.numpy()
        # pred = pred.numpy()
        return compute_eer(pred[target == 1], pred[target == 0])[0]
