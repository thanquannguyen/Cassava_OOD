import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_fpr95(id_scores, ood_scores):
    """
    Calculate False Positive Rate at 95% True Positive Rate.
    Assumes ID scores are LOWER than OOD scores for Energy (if using standard Energy).
    Wait, Energy = -T * logsumexp.
    High prob -> High logsumexp -> Low Energy (more negative).
    Low prob -> Low logsumexp -> High Energy (less negative).
    So ID has LOW Energy, OOD has HIGH Energy.
    
    We treat this as a binary classification: 0 for ID, 1 for OOD.
    """
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find FPR when TPR >= 0.95
    # Note: TPR is Recall (Sensitivity). We want 95% of ID samples to be accepted.
    # But here 0 is ID, 1 is OOD.
    # So "Positive" is OOD. "True Positive" is detecting OOD as OOD.
    # "False Positive" is detecting ID as OOD.
    # We want TPR (OOD detection rate) to be 95%? No, usually we fix TPR of ID to 95%.
    # Standard metric: FPR at 95% TPR.
    # Usually defined as: FPR of OOD (class 1) when TPR of ID (class 0) is 95%.
    # Let's stick to the standard definition:
    # FPR95 = Probability that an OOD example is misclassified as ID when 95% of ID examples are correctly classified.
    
    # Let's flip it to make it easier:
    # ID = 1, OOD = 0.
    # We want 95% TPR (ID correctly classified).
    # What is the FPR (OOD misclassified as ID)?
    
    # Re-doing with ID=1 (Positive), OOD=0 (Negative).
    # Score: Energy. ID has Low Energy. OOD has High Energy.
    # So we need to negate Energy to make ID have High Score.
    # Score = -Energy.
    
    scores_id = -np.array(id_scores)
    scores_ood = -np.array(ood_scores)
    
    y_true = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    y_scores = np.concatenate([scores_id, scores_ood])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find threshold where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0][0]
    return fpr[idx]

def calculate_auroc(id_scores, ood_scores):
    # ID = 0, OOD = 1.
    # Score = Energy.
    # OOD should have higher energy.
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    return roc_auc_score(y_true, y_scores)

def expected_calibration_error(confidences, predictions, labels, num_bins=15):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece
