from splitting_data import splitting_data
from keras.api.models import Model
from keras.api.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input, LSTM, Concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import numpy as np

def ensemble(dataframe, target_col):
    label_encoder = LabelEncoder().fit(dataframe[target_col])
    labels = label_encoder.transform(dataframe[target_col])
    classes = list(label_encoder.classes_)

    X = dataframe.drop(columns=[target_col], axis=1)
    y = labels

    if len(classes) == 2:
        loss_function = "binary_crossentropy"
        activation_function = "sigmoid"
        output_nodes = 1
    else:
        loss_function = "sparse_categorical_crossentropy"
        activation_function = "softmax"
        output_nodes = len(classes)

    X_train, y_train, X_test, y_test = splitting_data(X, y, train_size=0.8, random_state=42, require_val=False)

    n_features = X_train.shape[1]

    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define individual model architectures
    def create_dnn_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(100, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)  # Flatten the output
        return Model(inputs=inputs, outputs=x)

    def create_cnn_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        return Model(inputs=inputs, outputs=x)

    def create_rnn_model(input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0, return_sequences=False)(inputs)
        x = Dropout(0.2)(x)
        return Model(inputs=inputs, outputs=x)

    def create_birnn_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0, return_sequences=False))(inputs)
        x = Dropout(0.2)(x)
        return Model(inputs=inputs, outputs=x)

    # Create ensemble model
    def create_ensemble_model(models, input_shape):
        inputs = Input(shape=input_shape)
        outputs = [model(inputs) for model in models]
        ensemble_output = Concatenate()(outputs)
        ensemble_output = Dense(128, activation='relu')(ensemble_output)
        ensemble_output = Dense(output_nodes, activation=activation_function)(ensemble_output)
        model = Model(inputs=inputs, outputs=ensemble_output)
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        return model

    # Create individual models
    dnn_model = create_dnn_model((n_features, 1))
    cnn_model = create_cnn_model((n_features, 1))
    rnn_model = create_rnn_model((n_features, 1))
    birnn_model = create_birnn_model((n_features, 1))

    models = [dnn_model, cnn_model, rnn_model, birnn_model]

    # Create ensemble model
    ensemble_model = create_ensemble_model(models, (n_features, 1))

    # Train the ensemble model
    batch_size = 32
    epochs = 100
    ensemble_model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

    # Make predictions
    ensemble_predictions = ensemble_model.predict(X_test_reshaped)

    if len(classes) == 2:
        ensemble_predictions_labels = (ensemble_predictions > 0.5).astype(int)
    else:
        ensemble_predictions_labels = np.argmax(ensemble_predictions, axis=-1)

    # Calculate Loss
    ensemble_loss = log_loss(y_test, ensemble_predictions)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, ensemble_predictions_labels)

    # Calculate Specificity and AUC-ROC
    if len(classes) == 2:
        ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        return ensemble_loss, ensemble_roc_auc, tp, fp, tn, fn
    else:
        ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions, multi_class='ovr')

        # Initialize TP, FP, TN, FN for each class
        tp_per_class = []
        fp_per_class = []
        tn_per_class = []
        fn_per_class = []

        # Loop through each class to calculate TP, FP, TN, FN
        for i in range(len(classes)):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            tn = conf_matrix.sum() - (tp + fp + fn)

            tp_per_class.append(tp)
            fp_per_class.append(fp)
            tn_per_class.append(tn)
            fn_per_class.append(fn)

        return ensemble_loss, ensemble_roc_auc, tp_per_class, fp_per_class, tn_per_class, fn_per_class