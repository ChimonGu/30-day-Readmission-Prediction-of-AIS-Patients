import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization, GRU, Input,
                                     Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout, Reshape, Attention)
from tensorflow.keras.regularizers import l2
import os
import requests
from zipfile import ZipFile


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def balanced_accuracy(y_true, y_pred):
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    return (sens + spec) / 2.0

class Clinicalmodel(tf.keras.Model):
    def __init__(self, clinical_input_shape):
        super(Clinicalmodel, self).__init__()
        self.clinical_branch = tf.keras.Sequential([
            tf.keras.layers.Reshape((clinical_input_shape[0], 1), input_shape=clinical_input_shape),
            tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.clinical_branch(inputs)


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(in_channels // reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(in_channels, activation='sigmoid')

    def call(self, x):
        channel_wise_avg = self.global_avg_pooling(x)
        channel_wise_weights = self.fc2(self.fc1(channel_wise_avg))
        channel_wise_weights = tf.expand_dims(channel_wise_weights, axis=1)
        channel_wise_weights = tf.tile(channel_wise_weights, [1, tf.shape(x)[1], 1])
        output = x * channel_wise_weights
        return output


class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = tf.keras.layers.Conv1D(out_channels, kernel_size=3, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

    def call(self, x):
        x = self.pool(self.relu(self.batch_norm(self.conv(x))))
        return x


class ClinicalBranch(tf.keras.layers.Layer):
    def __init__(self, clinical_input_shape):
        super(ClinicalBranch, self).__init__()
        self.clinical_branch = tf.keras.Sequential([
            tf.keras.layers.Reshape((clinical_input_shape[0], 1), input_shape=(clinical_input_shape[0],)),
            tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=clinical_input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5)
        ])

    def call(self, clinical_inputs):
        clinical_features = self.clinical_branch(clinical_inputs)
        return clinical_features


class SEInceptionCNN(tf.keras.Model):
    def __init__(self, clinical_input_shape, ecg_input_shape, num_classes=1):
        super(SEInceptionCNN, self).__init__()

        # Branch for processing clinical data
        self.clinical_branch = ClinicalBranch(clinical_input_shape)

        # Branch for processing ECG data
        # Encoder
        self.autoencoder_conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=ecg_input_shape, padding='same')
        self.autoencoder_pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')
        self.autoencoder_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')
        self.autoencoder_pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')

        # Decoder
        self.autoencoder_conv3 = tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')
        self.autoencoder_upsample1 = tf.keras.layers.UpSampling1D(size=2)
        self.autoencoder_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')
        self.autoencoder_upsample2 = tf.keras.layers.UpSampling1D(size=2)
        self.autoencoder_conv5 = tf.keras.layers.Conv1D(filters=12, kernel_size=3, activation='linear', padding='same')

        self.squeeze_excitation = SqueezeExcitation(ecg_input_shape[-1])
        self.cnn_block1 = CNNBlock(ecg_input_shape[-1], 64)
        self.cnn_block2 = CNNBlock(64, 128)
        self.cnn_block3 = CNNBlock(128, 256)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Merge both branches
        self.concat = tf.keras.layers.Concatenate()

        # Final classification layer
        self.fc = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Split inputs into clinical and ECG data
        clinical_inputs, ecg_inputs = inputs

        # Process clinical data
        clinical_features = self.clinical_branch(clinical_inputs)

        # Process ECG data
        ecg_features = self.autoencoder_conv1(ecg_inputs)
        ecg_features = self.autoencoder_pool1(ecg_features)
        ecg_features = self.autoencoder_conv2(ecg_features)
        ecg_features = self.autoencoder_pool2(ecg_features)

        ecg_features = self.autoencoder_conv3(ecg_features)
        ecg_features = self.autoencoder_upsample1(ecg_features)
        ecg_features = self.autoencoder_conv4(ecg_features)
        ecg_features = self.autoencoder_upsample2(ecg_features)
        ecg_features = self.autoencoder_conv5(ecg_features)

        ecg_features = self.squeeze_excitation(ecg_features)
        ecg_features = self.cnn_block1(ecg_features)
        ecg_features = self.cnn_block2(ecg_features)
        ecg_features = self.cnn_block3(ecg_features)
        ecg_features = tf.transpose(ecg_features, perm=[0, 2, 1])
        ecg_features = self.bi_lstm(ecg_features)
        ecg_features = self.global_avg_pooling(ecg_features)
        ecg_features = self.dropout(ecg_features)

        # Concatenate or merge features from both branches
        merged_features = self.concat([clinical_features, ecg_features])

        # Final classification layer
        output = self.fc(merged_features)

        return output


class AutoencoderClassifierModel(Model):
    def __init__(self, ecg_input_shape):
        super(AutoencoderClassifierModel, self).__init__()

        # Encoder
        self.encoder = Sequential([
            Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=ecg_input_shape),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2, padding='same')
        ])

        # Decoder
        self.decoder = Sequential([
            Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'),
            UpSampling1D(size=2),
            Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
            UpSampling1D(size=2),
            Conv1D(filters=12, kernel_size=3, activation='linear', padding='same')
        ])

        # Classification Model
        self.classification_model = Sequential([
            Flatten(input_shape=(ecg_input_shape[0], ecg_input_shape[1])),
            BatchNormalization(),
            Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(units=32, activation='relu'),
            Dropout(0.5),
            Dense(units=1, activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        # Forward pass
        encoded_features = self.encoder(inputs)
        decoded_features = self.decoder(encoded_features)
        classification_output = self.classification_model(decoded_features)

        return classification_output


def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())


def preprocess_clinical_data(clinical_inputs):
    X_clinical = clinical_inputs.iloc[:, 1:]
    continual_val = ['last_24h_hematocrit_mean', 'last_24h_hemoglobin_mean']
    if X_clinical.shape[0] > 1:
        X_clinical[continual_val] = X_clinical[continual_val].apply(scale_minmax, axis=0)
    else:
        return X_clinical

    return X_clinical


def preprocess_ecg_data(ecg_inputs):
    grouped_data = ecg_inputs.groupby('subject_id')
    max_time_steps = 5000
    adjusted_data = np.zeros((len(grouped_data), max_time_steps, ecg_inputs.shape[1] - 1))
    scaler = StandardScaler()

    for i, (name, group) in enumerate(grouped_data):
        group_data = group.drop(columns=['subject_id'], axis=1).values
        group_data = scaler.fit_transform(group_data)
        if group_data.shape[0] > max_time_steps:
          group_data = group_data[:max_time_steps, :]
        adjusted_data[i, :len(group_data), :] = group_data

    X_ecg = adjusted_data

    return X_ecg


def make_prediction_combined(model, clinical_inputs, ecg_inputs):
    X_clinical = preprocess_clinical_data(clinical_inputs)
    X_ecg = preprocess_ecg_data(ecg_inputs)
    predictions = model.predict([X_clinical, X_ecg])

    return predictions


def make_prediction_clinical(model, clinical_inputs):
    X_clinical = preprocess_clinical_data(clinical_inputs)
    predictions = model.predict(X_clinical)

    return predictions


def make_prediction_ecg(model, ecg_inputs):
    X_ecg = preprocess_ecg_data(ecg_inputs)
    predictions = model.predict(X_ecg)

    return predictions


current_directory = os.path.dirname(os.path.abspath(__file__))
weights_filename = "saved_model_weights.zip"
weights_path = os.path.join(current_directory, weights_filename)
weights_url = "https://github.com/ChimonGu/30-day-Readmission-Prediction-of-AIS-Patients/releases/download/weights/saved_model_weights.zip"
absolute_extract_path = os.path.join(current_directory, "saved_model_weights")
response = requests.get(weights_url)
with open(weights_path, 'wb') as f:
    f.write(response.content)
with ZipFile(weights_path, 'r') as zip_ref:
    zip_ref.extractall(absolute_extract_path)


# Load model
model_combined = SEInceptionCNN(clinical_input_shape=(15, ), ecg_input_shape=(5000, 12), num_classes=1)
# model_combined.load_weights(r"C:\Users\chimo\Desktop\python_project\#2-paper\二分类\ecg+clinical\saved_model\multimodel_lasso_cnnae+seinception_weights")
model_combined.load_weights(os.path.join(absolute_extract_path, "saved_model_weights/multimodel_lasso_cnnae+seinception_weights"))

model_clinical = Clinicalmodel(clinical_input_shape=(15,))
# model_clinical.load_weights(r"C:\Users\chimo\Desktop\python_project\#2-paper\二分类\ecg+clinical\saved_model\clinicalbranch_weights")
model_clinical.load_weights(os.path.join(absolute_extract_path, "saved_model_weights/clinicalbranch_weights"))

model_ecg = AutoencoderClassifierModel(ecg_input_shape=(5000, 12))
# model_ecg.load_weights(r"C:\Users\chimo\Desktop\python_project\#2-paper\二分类\ecg\saved_model\cnnae_weights")
model_ecg.load_weights(os.path.join(absolute_extract_path, "saved_model_weights/cnnae_weights"))


# Streamlit app 
st.title("30-Day Readmission Prediction of AIS Patients")

# Add input elements
uploaded_file = st.file_uploader("Upload ECG file", type=["csv"])
clinical_data = st.file_uploader("Upload Clinical file", type=["csv"])

if uploaded_file is not None or clinical_data is not None:

    if uploaded_file is not None and clinical_data is not None:

        ecg_data = pd.read_csv(uploaded_file)
        Clinical_data = pd.read_csv(clinical_data)

        if st.button("Predict"):
            # Use model3 for prediction
            result = make_prediction_combined(model_combined, Clinical_data, ecg_data)
            result_df = pd.DataFrame(result, columns=['Readmission Probability'])
            result_df['Risk Level'] = pd.cut(result_df['Readmission Probability'], bins=[0, 0.2, 0.8, 1],
                                             labels=['Low Risk', 'Medium Risk', 'High Risk'])
            new_order = ['Readmission Probability', 'Risk Level']
            result_df = result_df[new_order]
            st.dataframe(result_df)

    elif uploaded_file is not None:

        ecg_data = pd.read_csv(uploaded_file)

        if st.button("Predict"):
            result = make_prediction_ecg(model_ecg, ecg_data)
            result_df = pd.DataFrame(result, columns=['Readmission Probability'])
            result_df['Risk Level'] = pd.cut(result_df['Readmission Probability'], bins=[0, 0.2, 0.8, 1],
                                             labels=['Low Risk', 'Medium Risk', 'High Risk'])
            new_order = ['Readmission Probability', 'Risk Level']
            result_df = result_df[new_order]
            st.dataframe(result_df)

    elif clinical_data is not None:

        Clinical_data = pd.read_csv(clinical_data)

        if st.button("Predict"):
            result = make_prediction_clinical(model_clinical, Clinical_data)
            result_df = pd.DataFrame(result, columns=['Readmission Probability'])
            result_df['Risk Level'] = pd.cut(result_df['Readmission Probability'],
                                             bins=[0, 0.2, 0.8, 1],
                                             labels=['Low Risk', 'Medium Risk', 'High Risk'],
                                             include_lowest=True)
            new_order = ['Readmission Probability', 'Risk Level']
            result_df = result_df[new_order]
            st.dataframe(result_df)









