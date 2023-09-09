from sklearn.metrics import make_scorer, precision_score
import numpy as np

def precision_at_full_recall(y_true, y_pred_probas, return_thd=False):
    # Sort samples by predicted probabilities
    sorted_indices = np.argsort(y_pred_probas)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # Find the position where the last true positive sample appears
    last_true_positive_index = len(y_true_sorted) - list(y_true_sorted[::-1]).index(1) - 1
    
    # Create predictions: all samples up to the last_true_positive_index are predicted as 1, rest as 0
    y_pred = [1 if i <= last_true_positive_index else 0 for i in range(len(y_pred_probas))]    
    if not return_thd:
        return precision_score(y_true_sorted, y_pred)
    else:
        probas_sorted = y_pred_probas[sorted_indices]
        return precision_score(y_true_sorted, y_pred), probas_sorted[last_true_positive_index]

def test_precision_at_full_recall():
    # Test Case 1
    y_true1 = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    y_pred_probas1 = [0.8, 0.4, 0.7, 0.3, 0.2, 0.5, 0.9, 0.6, 0.1, 0.05]
    assert precision_at_full_recall(y_true1, y_pred_probas1) == 1.0  # expected: 100% precision
    
    # Test Case 2: case where positives are interspersed with negatives
    y_true2 = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    y_pred_probas2 = [0.8, 0.4, 0.7, 0.3, 0.65, 0.5, 0.9, 0.6, 0.1, 0.05]
    assert precision_at_full_recall(y_true2, y_pred_probas2) == 3/4  # expected: 50% precision
    
    # Test Case 3: case where all samples are positive
    y_true3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred_probas3 = [0.8, 0.4, 0.7, 0.3, 0.65, 0.5, 0.9, 0.6, 0.1, 0.05]
    assert precision_at_full_recall(y_true3, y_pred_probas3) == 1.0  # expected: 100% precision
    
    print("All tests passed!")
    
    
precision_at_full_recall_scorer = make_scorer(precision_at_full_recall, needs_proba=True)