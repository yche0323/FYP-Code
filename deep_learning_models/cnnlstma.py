from sklearn.metrics import roc_auc_score, confusion_matrix
from keras.api.models import Sequential
from keras.api.layers import Conv1D, Activation, MaxPooling1D, LSTM, Flatten, BatchNormalization, Dropout, Dense, Input
from keras.api.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import numpy as np
from splitting_data import splitting_data
from deep_learning_models.metrics import weighted_metrics

def cnnlstma(dataframe, target_col, neuron1=2048, neuron2=1024, batch_size=32, dropout_rate=0.15):
    # Prepare the data
    label_encoder = LabelEncoder().fit(dataframe[target_col])
    labels = label_encoder.transform(dataframe[target_col])
    classes = list(label_encoder.classes_)
    num_classes = len(classes)

    X = dataframe.drop(columns=[target_col], axis=1)
    y = labels

    # Split data into train and test sets
    X_train, y_train, X_test, y_test = splitting_data(X, y, train_size=0.8, random_state=42, require_val=False)
    
    # Standardise the data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape the input data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Conv1D(500, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(250, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(neuron1, activation="relu"))  # neuron 1
    model.add(Dense(neuron2, activation="relu"))  # neuron 2
    model.add(Dense(len(classes), activation="softmax"))

    # Define a learning rate schedule
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )

    # Use the learning rate schedule in the optimiser
    opt = SGD(learning_rate=lr_schedule, momentum=0.3, nesterov=True)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=[])

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    cnnlstma_loss = model.evaluate(X_test, y_test, verbose=0)
    cnnlstma_predictions = model.predict(X_test, verbose=0)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(classes))
    roc_auc = roc_auc_score(y_test_one_hot, cnnlstma_predictions, average="macro", multi_class="ovr")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, np.argmax(cnnlstma_predictions, axis=1))

    if num_classes == 2:
        tn, fp, fn, tp = conf_matrix.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

        return round(cnnlstma_loss, 4), round(acc * 100, 2), round(prec * 100, 2), round(sens * 100, 2), round(spec * 100, 2), round(f1 * 100, 2), round(roc_auc * 100, 2)
    else:
        # Initialize TP, FP, TN, FN for each class
        tp_per_class = []
        fp_per_class = []
        tn_per_class = []
        fn_per_class = []

        # Loop through each class to calculate TP, FP, TN, FN
        for i in range(len(classes)):
            tp = conf_matrix[i, i]  # True Positive is the diagonal element
            fp = conf_matrix[:, i].sum() - tp  # False Positive is column sum minus the diagonal element
            fn = conf_matrix[i, :].sum() - tp  # False Negative is row sum minus the diagonal element
            tn = conf_matrix.sum() - (tp + fp + fn)  # True Negative is everything else

            tp_per_class.append(tp)
            fp_per_class.append(fp)
            tn_per_class.append(tn)
            fn_per_class.append(fn)

        acc, prec, sens, spec, f1 = weighted_metrics(num_classes, tp_per_class, fp_per_class, tn_per_class, fn_per_class)

        return round(cnnlstma_loss, 4), round(acc * 100, 2), round(prec * 100, 2), round(sens * 100, 2), round(spec * 100, 2), round(f1 * 100, 2), round(roc_auc * 100, 2)
