import os
from datetime import datetime
from data_tool_box import *
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def set_up_exp_folder(cwd,exp = 'experiment_logs/'):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = cwd + exp
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True,
                         sample_weight=None, labels=None):
    """Top-k Accuracy classification score.
    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.
    Read more in the :ref:`User Guide <top_k_accuracy_score>`
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers). The binary case expects
        scores with shape (n_samples,) while the multiclass case expects scores
        with shape (n_samples, n_classes). In the nulticlass case, the order of
        the class scores must correspond to the order of ``labels``, if
        provided, or else to the numerical or lexicographical order of the
        labels in ``y_true``.
    k : int, default=2
        Number of most likely outcomes considered to find the correct label.
    normalize : bool, default=True
        If `True`, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.
    labels : array-like of shape (n_classes,), default=None
        Multiclass only. List of labels that index the classes in ``y_score``.
        If ``None``, the numerical or lexicographical order of the labels in
        ``y_true`` is used.
    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.
    See also
    --------
    accuracy_score
    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.
    Examples
    --------
    # >>> import numpy as np
    # >>> from sklearn.metrics import top_k_accuracy_score
    # >>> y_true = np.array([0, 1, 2, 2])
    # >>> y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
    # ...                     [0.3, 0.4, 0.2],  # 1 is in top 2
    # ...                     [0.2, 0.4, 0.3],  # 2 is in top 2
    # ...                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
    # >>> top_k_accuracy_score(y_true, y_score, k=2)
    # 0.75
    # >>> # Not normalizing gives the number of "correctly" classified samples
    # >>> top_k_accuracy_score(y_true, y_score, k=2, normalize=False)
    # 3
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_type = type_of_target(y_true)
    y_score = check_array(y_score, ensure_2d=False)
    y_score = column_or_1d(y_score) if y_type == 'binary' else y_score
    check_consistent_length(y_true, y_score, sample_weight)

    if y_type not in {'binary', 'multiclass'}:
        raise ValueError(
            f"y type must be 'binary' or 'multiclass', got '{y_type}' instead."
        )

    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    if labels is None:
        classes = _unique(y_true)
        n_classes = len(classes)

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of classes in 'y_true' ({n_classes}) not equal "
                f"to the number of classes in 'y_score' ({y_score_n_classes})."
            )
    else:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)

        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")

        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of given labels ({n_classes}) not equal to the "
                f"number of classes in 'y_score' ({y_score_n_classes})."
            )

        if len(np.setdiff1d(y_true, classes)):
            raise ValueError(
                "'y_true' contains labels not in parameter 'labels'."
            )

    if k >= n_classes:
        warnings.warn(
            f"'k' ({k}) greater than or equal to 'n_classes' ({n_classes}) "
            "will result in a perfect score and is therefore meaningless.",
            UndefinedMetricWarning
        )

    y_true_encoded = _encode(y_true, uniques=classes)

    if y_type == 'binary':
        if k == 1:
            threshold = .5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif y_type == 'multiclass':
        sorted_pred = np.argsort(y_score, axis=1, kind='mergesort')[:, ::-1]
        hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    if normalize:
        return np.average(hits, weights=sample_weight)
    elif sample_weight is None:
        return np.sum(hits)
    else:
        return np.dot(hits, sample_weight)

