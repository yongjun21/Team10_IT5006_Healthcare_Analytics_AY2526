from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

import statsmodels.stats.api as sms

import time
import requests
import pickle


def load_data():
    # return fetch_ucirepo(id=296)
    response = requests.get('https://assets.yongjun.sg/diabetes_dataset.pkl')
    dataset = pickle.loads(response.content)
    return dataset


def evaluate_model(get_model, train_X, train_y, test_X, test_y, get_decision_score=None):
    model = get_model()

    start_time = time.time()
    model.fit(train_X, train_y)
    training_time = time.time() - start_time

    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    train_accuracy = accuracy_score(train_y, train_pred)
    test_accuracy = accuracy_score(test_y, test_pred)

    # Calculate average metrics (macro average)
    train_avg_precision = precision_score(train_y, train_pred, average='macro')
    test_avg_precision = precision_score(test_y, test_pred, average='macro')
    train_avg_recall = recall_score(train_y, train_pred, average='macro')
    test_avg_recall = recall_score(test_y, test_pred, average='macro')
    train_avg_f1 = f1_score(train_y, train_pred, average='macro')
    test_avg_f1 = f1_score(test_y, test_pred, average='macro')
    
    # Calculate target class metrics (class 1)
    train_target_precision = precision_score(train_y, train_pred, pos_label=1)
    test_target_precision = precision_score(test_y, test_pred, pos_label=1)
    train_target_recall = recall_score(train_y, train_pred, pos_label=1)
    test_target_recall = recall_score(test_y, test_pred, pos_label=1)
    train_target_f1 = f1_score(train_y, train_pred, pos_label=1)
    test_target_f1 = f1_score(test_y, test_pred, pos_label=1)

    results = {
        "model": model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "overfitting_accuracy": train_accuracy - test_accuracy,
        "train_precision_score": precision_score(train_y, train_pred),
        "test_precision_score": precision_score(test_y, test_pred),
        "train_recall_score": recall_score(train_y, train_pred),
        "test_recall_score": recall_score(test_y, test_pred),
        "train_f1": f1_score(train_y, train_pred),
        "test_f1": f1_score(test_y, test_pred),
        # Average metrics
        "train_avg_precision": train_avg_precision,
        "test_avg_precision": test_avg_precision,
        "train_avg_recall": train_avg_recall,
        "test_avg_recall": test_avg_recall,
        "train_avg_f1": train_avg_f1,
        "test_avg_f1": test_avg_f1,
        # Target class metrics
        "train_target_precision": train_target_precision,
        "test_target_precision": test_target_precision,
        "train_target_recall": train_target_recall,
        "test_target_recall": test_target_recall,
        "train_target_f1": train_target_f1,
        "test_target_f1": test_target_f1,
        "train_cm": confusion_matrix(train_y, train_pred),
        "test_cm": confusion_matrix(test_y, test_pred),
        "training_time": training_time
    }

    if get_decision_score is not None:
        train_decision_score = get_decision_score(model, train_X)
        test_decision_score = get_decision_score(model, test_X)

        # Calculate ROC curves
        train_fpr, train_tpr, train_thresholds = roc_curve(train_y, train_decision_score)
        test_fpr, test_tpr, test_thresholds = roc_curve(test_y, test_decision_score)

        # Calculate precision-recall curves
        train_precision, train_recall, train_pr_thresholds = precision_recall_curve(train_y, train_decision_score)
        test_precision, test_recall, test_pr_thresholds = precision_recall_curve(test_y, test_decision_score)

        results["train_decision_score"] = train_decision_score
        results["test_decision_score"] = test_decision_score
        results["train_roc_auc"] = auc(train_fpr, train_tpr)
        results["test_roc_auc"] = auc(test_fpr, test_tpr)
        results["overfitting_auc"] = results["train_roc_auc"] - results["test_roc_auc"]
        results["train_pr_auc"] = auc(train_recall, train_precision)
        results["test_pr_auc"] = auc(test_recall, test_precision)
        
        # Calculate PR AUC for target class (class 1)
        # For target class, we use the decision scores directly
        train_target_pr_auc = auc(train_recall, train_precision)
        test_target_pr_auc = auc(test_recall, test_precision)
        
        # For average PR AUC, we need to calculate for both classes and average
        # Calculate for null class (class 0) by inverting the decision scores
        train_y_null = 1 - train_y
        test_y_null = 1 - test_y
        train_decision_score_null = 1 - train_decision_score
        test_decision_score_null = 1 - test_decision_score
        
        train_precision_null, train_recall_null, _ = precision_recall_curve(train_y_null, train_decision_score_null)
        test_precision_null, test_recall_null, _ = precision_recall_curve(test_y_null, test_decision_score_null)
        
        train_pr_auc_null = auc(train_recall_null, train_precision_null)
        test_pr_auc_null = auc(test_recall_null, test_precision_null)
        
        # Average PR AUC
        train_avg_pr_auc = (train_target_pr_auc + train_pr_auc_null) / 2
        test_avg_pr_auc = (test_target_pr_auc + test_pr_auc_null) / 2
        
        results["train_target_pr_auc"] = train_target_pr_auc
        results["test_target_pr_auc"] = test_target_pr_auc
        results["train_avg_pr_auc"] = train_avg_pr_auc
        results["test_avg_pr_auc"] = test_avg_pr_auc
        
        results["train_roc_curve"] = train_fpr, train_tpr
        results["test_roc_curve"] = test_fpr, test_tpr
        results["train_pr_curve"] = train_precision, train_recall
        results["test_pr_curve"] = test_precision, test_recall

    return results


def cv_evaluate_model(get_model, train_Xs, train_ys, test_Xs, test_ys, get_decision_score=None):
    results = []
    train_accuracy = []
    test_accuracy = []
    overfitting_accuracy = []
    train_precision_score = []
    test_precision_score = []
    train_recall_score = []
    test_recall_score = []
    train_f1 = []
    test_f1 = []
    # Average metrics
    train_avg_precision = []
    test_avg_precision = []
    train_avg_recall = []
    test_avg_recall = []
    train_avg_f1 = []
    test_avg_f1 = []
    # Target class metrics
    train_target_precision = []
    test_target_precision = []
    train_target_recall = []
    test_target_recall = []
    train_target_f1 = []
    test_target_f1 = []
    training_time = []
    train_roc_auc = []
    test_roc_auc = []
    overfitting_auc = []
    train_pr_auc = []
    test_pr_auc = []
    # New PR AUC metrics
    train_target_pr_auc = []
    test_target_pr_auc = []
    train_avg_pr_auc = []
    test_avg_pr_auc = []

    compiled = {}

    fold = 0
    for train_X, train_y, test_X, test_y in zip(train_Xs, train_ys, test_Xs, test_ys):
        result = evaluate_model(get_model, train_X, train_y, test_X, test_y, get_decision_score)
        print(f"Trained fold {fold} in {result['training_time']:.2f}s")
        fold += 1

        results.append(result)
        train_accuracy.append(result["train_accuracy"])
        test_accuracy.append(result["test_accuracy"])
        overfitting_accuracy.append(result["overfitting_accuracy"])
        train_precision_score.append(result["train_precision_score"])
        test_precision_score.append(result["test_precision_score"])
        train_recall_score.append(result["train_recall_score"])
        test_recall_score.append(result["test_recall_score"])
        train_f1.append(result["train_f1"])
        test_f1.append(result["test_f1"])
        # Average metrics
        train_avg_precision.append(result["train_avg_precision"])
        test_avg_precision.append(result["test_avg_precision"])
        train_avg_recall.append(result["train_avg_recall"])
        test_avg_recall.append(result["test_avg_recall"])
        train_avg_f1.append(result["train_avg_f1"])
        test_avg_f1.append(result["test_avg_f1"])
        # Target class metrics
        train_target_precision.append(result["train_target_precision"])
        test_target_precision.append(result["test_target_precision"])
        train_target_recall.append(result["train_target_recall"])
        test_target_recall.append(result["test_target_recall"])
        train_target_f1.append(result["train_target_f1"])
        test_target_f1.append(result["test_target_f1"])
        training_time.append(result["training_time"])
        if result.get("train_decision_score") is not None:
            train_roc_auc.append(result["train_roc_auc"])
            test_roc_auc.append(result["test_roc_auc"])
            overfitting_auc.append(result["overfitting_auc"])
            train_pr_auc.append(result["train_pr_auc"])
            test_pr_auc.append(result["test_pr_auc"])
            # New PR AUC metrics
            train_target_pr_auc.append(result["train_target_pr_auc"])
            test_target_pr_auc.append(result["test_target_pr_auc"])
            train_avg_pr_auc.append(result["train_avg_pr_auc"])
            test_avg_pr_auc.append(result["test_avg_pr_auc"])

    compiled["results"] = results
    compiled["train_accuracy"] = get_metric_stats(train_accuracy)
    compiled["test_accuracy"] = get_metric_stats(test_accuracy)
    compiled["overfitting_accuracy"] = get_metric_stats(overfitting_accuracy)
    compiled["train_precision_score"] = get_metric_stats(train_precision_score)
    compiled["test_precision_score"] = get_metric_stats(test_precision_score)
    compiled["train_recall_score"] = get_metric_stats(train_recall_score)
    compiled["test_recall_score"] = get_metric_stats(test_recall_score)
    compiled["train_f1"] = get_metric_stats(train_f1)
    compiled["test_f1"] = get_metric_stats(test_f1)
    # Average metrics
    compiled["train_avg_precision"] = get_metric_stats(train_avg_precision)
    compiled["test_avg_precision"] = get_metric_stats(test_avg_precision)
    compiled["train_avg_recall"] = get_metric_stats(train_avg_recall)
    compiled["test_avg_recall"] = get_metric_stats(test_avg_recall)
    compiled["train_avg_f1"] = get_metric_stats(train_avg_f1)
    compiled["test_avg_f1"] = get_metric_stats(test_avg_f1)
    # Target class metrics
    compiled["train_target_precision"] = get_metric_stats(train_target_precision)
    compiled["test_target_precision"] = get_metric_stats(test_target_precision)
    compiled["train_target_recall"] = get_metric_stats(train_target_recall)
    compiled["test_target_recall"] = get_metric_stats(test_target_recall)
    compiled["train_target_f1"] = get_metric_stats(train_target_f1)
    compiled["test_target_f1"] = get_metric_stats(test_target_f1)
    compiled["training_time"] = get_metric_stats(training_time)
    if get_decision_score is not None:
        compiled["train_roc_auc"] = get_metric_stats(train_roc_auc)
        compiled["test_roc_auc"] = get_metric_stats(test_roc_auc)
        compiled["overfitting_auc"] = get_metric_stats(overfitting_auc)
        compiled["train_pr_auc"] = get_metric_stats(train_pr_auc)
        compiled["test_pr_auc"] = get_metric_stats(test_pr_auc)
        # New PR AUC metrics
        compiled["train_target_pr_auc"] = get_metric_stats(train_target_pr_auc)
        compiled["test_target_pr_auc"] = get_metric_stats(test_target_pr_auc)
        compiled["train_avg_pr_auc"] = get_metric_stats(train_avg_pr_auc)
        compiled["test_avg_pr_auc"] = get_metric_stats(test_avg_pr_auc)

    return compiled


def get_metric_stats(data):
    lb, ub = sms.DescrStatsW(data).tconfint_mean()
    return 0.5 * lb + 0.5 * ub, (lb, ub)
