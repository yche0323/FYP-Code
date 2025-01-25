import tensorflow as tf

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())

    return precision

def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    return recall

def f1(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon())

    return f1

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * y_pred, 0, 1)))
    specificity_val = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())

    return specificity_val

def weighted_metrics(num_classes, tp_per_class, fp_per_class, tn_per_class, fn_per_class):
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    f1_score = []

    # Support for each class (total instances per class)
    support = [tp_per_class[i] + fn_per_class[i] for i in range(num_classes)]
    total_support = sum(support)

    # Per-class metrics
    for i in range(num_classes):
        total = tp_per_class[i] + fp_per_class[i] + tn_per_class[i] + fn_per_class[i]
        acc = (tp_per_class[i] + tn_per_class[i]) / total if total > 0 else 0
        prec = tp_per_class[i] / (tp_per_class[i] + fp_per_class[i]) if (tp_per_class[i] + fp_per_class[i]) > 0 else 0
        sens = tp_per_class[i] / (tp_per_class[i] + fn_per_class[i]) if (tp_per_class[i] + fn_per_class[i]) > 0 else 0
        spec = tn_per_class[i] / (tn_per_class[i] + fp_per_class[i]) if (tn_per_class[i] + fp_per_class[i]) > 0 else 0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0

        accuracy.append(acc)
        precision.append(prec)
        sensitivity.append(sens)
        specificity.append(spec)
        f1_score.append(f1)

    # Weighted metrics
    weighted_accuracy = sum(accuracy[i] * support[i] for i in range(num_classes)) / total_support
    weighted_precision = sum(precision[i] * support[i] for i in range(num_classes)) / total_support
    weighted_sensitivity = sum(sensitivity[i] * support[i] for i in range(num_classes)) / total_support
    weighted_specificity = sum(specificity[i] * support[i] for i in range(num_classes)) / total_support
    weighted_f1_score = sum(f1_score[i] * support[i] for i in range(num_classes)) / total_support

    return weighted_accuracy, weighted_precision, weighted_sensitivity, weighted_specificity, weighted_f1_score