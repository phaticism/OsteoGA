from sklearn.exceptions import InconsistentVersionWarning
import warnings
import gc
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import base64
from tensorflow.keras.applications import (
    Xception, EfficientNetV2S
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Lambda,
    Dropout, Input, concatenate, add, Conv2DTranspose,
)
from tensorflow.keras.optimizers import Adam
import absl.logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

absl.logging.set_verbosity(absl.logging.ERROR)


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


c = tf.constant([
    [1, 3, 6, 7, 9],
    [4, 1, 4, 5, 7],
    [6, 4, 1, 3, 5],
    [9, 7, 4, 1, 4],
    [11, 9, 7, 5, 1]
], dtype=tf.float32)

NUM_CLASSES = 44

BACKBONES = {
    'xception': Xception,
    'efficientnetv2s': EfficientNetV2S
}

WORKING_DIR = 'clinical/'

clinical_file = os.path.join(WORKING_DIR, 'clinical-2.csv')
clinical = pd.read_csv(clinical_file)
del clinical['ID']
del clinical['SIDE']

# columns_to_scale = ['AGE', 'HEIGHT', 'WEIGHT', 'MAX WEIGHT', 'BMI', 'KOOS PAIN SCORE']
# clinical[columns_to_scale] = scaler.fit_transform(clinical[columns_to_scale])

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', MinMaxScaler(), ['AGE', 'MAX WEIGHT', 'BMI'])
    ],
    remainder='passthrough',
    n_jobs=-1,
)

columns_to_convert = ['FREQUENT PAIN', 'SURGERY', 'RISK',
                      'SXKOA', 'SWELLING', 'BENDING FULLY', 'SYMPTOMATIC', 'CREPITUS']
mapping_dict = {}

for column in columns_to_convert:
    clinical[column], unique_values = pd.factorize(clinical[column])
    mapping_dict[column] = unique_values

train_dir = os.path.join(WORKING_DIR, 'original/df_train.csv')
df_train = pd.read_csv(train_dir).dropna()

test_dir = os.path.join(WORKING_DIR, 'original/df_test.csv')
df_test = pd.read_csv(test_dir).dropna()

val_dir = os.path.join(WORKING_DIR, 'original/df_val.csv')
df_val = pd.read_csv(val_dir).dropna()

df_train = df_train.merge(clinical, left_on='filename', right_on='FILENAME',
                          how='left').drop(columns=['FILENAME']).dropna()
df_test = df_test.merge(clinical, left_on='filename', right_on='FILENAME',
                        how='left').drop(columns=['FILENAME']).dropna()
df_val = df_val.merge(clinical, left_on='filename', right_on='FILENAME',
                      how='left').drop(columns=['FILENAME']).dropna()

X_train = df_train.iloc[:, 3:]
y_train = df_train['kl_grade']

X_val = df_val.iloc[:, 3:]
y_val = df_val['kl_grade']

X_test = df_test.iloc[:, 3:]
y_test = df_test['kl_grade']

X_train_np = df_train.iloc[:, 3:].to_numpy()
y_train_np = df_train['kl_grade'].to_numpy()

X_val_np = df_val.iloc[:, 3:].to_numpy()
y_val_np = df_val['kl_grade'].to_numpy()

X_test_np = df_test.iloc[:, 3:].to_numpy()
y_test_np = df_test['kl_grade'].to_numpy()

X_cv = pd.concat([df_train.iloc[:, 3:], df_val.iloc[:, 3:]])
y_cv = pd.concat([df_train.iloc[:, 1], df_val.iloc[:, 1]])

X_cv_np = X_cv.to_numpy()
y_cv_np = y_cv.to_numpy()


def adjust_pretrained_weights(model_cls, input_size, name=None):
    weights_model = model_cls(weights='imagenet',
                              include_top=False,
                              input_shape=(*input_size, 3))
    target_model = model_cls(weights=None,
                             include_top=False,
                             input_shape=(*input_size, 1))
    weights = weights_model.get_weights()
    weights[0] = np.sum(weights[0], axis=2, keepdims=True)
    target_model.set_weights(weights)

    del weights_model
    tf.keras.backend.clear_session()
    gc.collect()
    if name:
        target_model._name = name
    return target_model


def get_adjusted_backbone(backbone='xception',
                          backbone_weights=None,
                          backbone_trainable=True,
                          input_shape=(224, 224, 1)):
    if backbone_weights == 'imagenet':
        if input_shape[-1] != 3:
            backbone_layer = adjust_pretrained_weights(
                BACKBONES[backbone], input_shape[:-1])
        else:
            backbone_layer = BACKBONES[backbone](weights=backbone_weights,
                                                 include_top=False,
                                                 input_shape=input_shape)
    elif backbone_weights is None:
        backbone_layer = BACKBONES[backbone](weights=backbone_weights,
                                             include_top=False,
                                             input_shape=input_shape)
    else:
        backbone_layer = BACKBONES[backbone](weights=None,
                                             include_top=False,
                                             input_shape=input_shape)
        backbone_layer.set_weights(backbone_weights)

    if not backbone_trainable:
        backbone_layer.trainable = False

    return backbone_layer


class Classifier:
    def __init__(self, svc_path, cnn_weights_path):
        self.backbone = get_adjusted_backbone(backbone='efficientnetv2s',
                                              backbone_weights='imagenet',
                                              backbone_trainable=True,
                                              input_shape=(224, 224, 1)
                                              )
        self.svc_path = svc_path
        self.cnn_weights_path = cnn_weights_path
        self.svc = self._load_svc()
        self.cnn_model = self._build_cnn_model()
        self.osteophytes = ['osteophytes_def',
                            'osteophytes_none', 'osteophytes_poss']
        self.jsn = ['jsn_def', 'jsn_mild/mod', 'jsn_none', 'jsn_severe']
        self.names = [
            'kl_grade_0', 'kl_grade_1', 'kl_grade_2', 'kl_grade_3', 'kl_grade_4',
            'osteophytes_def', 'osteophytes_none', 'osteophytes_poss',
            'jsn_def', 'jsn_mild/mod', 'jsn_none', 'jsn_severe',
            'osfl_0', 'osfl_1', 'osfl_2', 'osfl_3',
            'scfl_0', 'scfl_1', 'scfl_2', 'scfl_3',
            'ostm_0', 'ostm_1', 'ostm_2', 'ostm_3',
            'sctm_0', 'sctm_1', 'sctm_2', 'sctm_3',
            'osfm_0', 'osfm_1', 'osfm_2', 'osfm_3',
            'scfm_0', 'scfm_1', 'scfm_2', 'scfm_3',
            'ostl_0', 'ostl_1', 'ostl_2', 'ostl_3',
            'sctl_0', 'sctl_1', 'sctl_2', 'sctl_3',
            'AGE', 'MAX WEIGHT', 'BMI',
        ]

        self.features = [
            'kl_grade_0', 'kl_grade_1', 'kl_grade_2', 'kl_grade_3', 'kl_grade_4',
            'osteophytes_def', 'osteophytes_poss',
            'jsn_mild/mod', 'jsn_none', 'jsn_severe',
            'osfl_0', 'osfl_1', 'scfl_0', 'scfl_1', 'ostm_2',
            'sctm_0', 'sctm_2', 'osfm_3', 'scfm_0', 'scfm_1',
            'scfm_3', 'ostl_1', 'sctl_0', 'sctl_3',
            'AGE', 'MAX WEIGHT', 'BMI',
        ]

        from lime.lime_tabular import LimeTabularExplainer

        self.class_names = ['0', '1', '2', '3', '4']
        self.explainer = LimeTabularExplainer(
            X_cv[self.features].to_numpy(),
            feature_names=self.features,
            training_labels=y_cv_np,
            mode='classification',
            class_names=self.class_names,
            discretize_continuous=True,
            discretizer='quartile',
            sample_around_instance=True,
            random_state=1024,
        )

    def _load_svc(self):
        with open(self.svc_path, 'rb') as f:
            return pickle.load(f)

    def _df_to_list(self, input_df):
        return np.squeeze(input_df.to_numpy()).tolist()

    def _build_cnn_model(self):
        inputs = Input(shape=(224, 224, 1))
        img_ft = self.backbone(inputs)
        gpooling = GlobalAveragePooling2D()(img_ft)
        output = Dense(NUM_CLASSES, activation='sigmoid')(gpooling)

        MODEL_NAME = 'efficientnetv2s-subset'
        model = Model(inputs, output, name=MODEL_NAME)

        loss_func = 'binary_crossentropy'
        optimizer = Adam(learning_rate=3e-5)
        model.compile(optimizer=optimizer, loss=loss_func,
                      metrics=['binary_accuracy'])

        model.load_weights(self.cnn_weights_path)
        return model

    def predict_proba(self, img, age, max_weight, bmi):
        img = cv2.resize(img, (224, 224)) / 255.
        img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

        clinical = np.array([age, max_weight, bmi])

        pred = np.squeeze(self.cnn_model.predict(img))

        vector = np.expand_dims(np.concatenate([pred, clinical]), axis=0)
        y_pred_df = pd.DataFrame(vector, columns=self.names)

        kl_grade = np.squeeze(self.svc.predict_proba(
            y_pred_df[self.features])).tolist()

        dataaa = np.squeeze(y_pred_df[self.features].values.reshape(-1, 1))
        explanation_image_html = self.explain_and_get_image(dataaa)

        return {
            "kl_grade": kl_grade,
            "osteophytes": self._df_to_list(y_pred_df[self.osteophytes]),
            "jsn": self._df_to_list(y_pred_df[self.jsn]),
        }, explanation_image_html

    def explain_and_get_image(self, input_data):
        # Generate the explanation
        exp = self.explainer.explain_instance(
            input_data,
            predict_fn=self.predict_proba_df,
            num_features=10,
            top_labels=1,
        )
        exp.save_to_file('explanation.html')

        with open("explanation.html", "r") as html_file:
            html_content = html_file.read()
        os.remove("explanation.html")

        return html_content

    def predict_proba_df(self, input_data):
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)  # Reshape to 2D array
            input_data = pd.DataFrame(input_data, columns=self.features)
        return self.svc.predict_proba(input_data)


# classifier = Classifier('weights_5cls/svc.pkl',
#                         'weights_5cls/efficientnetv2s-subset.weights.h5')

# img = cv2.imread('archive/train/0/9007904L.png')

# # convert to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(classifier.predict_proba(img, 50, 100, 20))
