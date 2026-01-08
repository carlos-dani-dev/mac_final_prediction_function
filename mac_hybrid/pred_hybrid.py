from facenet_pytorch import MTCNN
import torch
from PIL import Image
import io
import sys
import cv2 as cv
import pandas as pd
import numpy as np
from functools import wraps
from keras_facenet import FaceNet
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


tf.config.set_visible_devices([], 'GPU')
print("[INFO] TensorFlow forçado a usar CPU.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device}")

CC_ANNO = pd.read_csv('./video_att_mapping_cc_dataset.csv').drop(columns=['Unnamed: 0'], axis=1)
ATTRIBUTES = CC_ANNO.columns[1:]
ATRIBUTOS_CC = np.append(ATTRIBUTES, 'male')
MODEL_HYBRID = load_model("./modelo_hybrid_200_2048_1e-05.keras")
MTCNN = MTCNN(keep_all=True, device=device)

LOADED_LABEL_ENCODERS = {}
with open('./label_encoders.pkl', 'rb') as f:
    LOADED_LABEL_ENCODERS = pickle.load(f)
FACENET = FaceNet()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def one_class_media_ponderada(one_class_predictions, a_factor):
    n = len(one_class_predictions)
    
    pred_sum = 0
    for pred in one_class_predictions:
        pred_sum+=pred
    return ((1-a_factor) / n) * pred_sum


def one_class_abs_diff_media(one_class_predictions, a_factor):
    n = len(one_class_predictions)

    pred_diff_sum = 0
    for pred_i in one_class_predictions:
        for pred_j in one_class_predictions:
            diff = abs(pred_i - pred_j)
            pred_diff_sum+=diff

    return (a_factor / (n * n)) * pred_diff_sum


def capture_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

    return wrapper


@capture_output
def get_embeddings(img):
    return FACENET.embeddings(img)


def get_image_prediction(img, training_or_not, model):
    img = extract_face(img, required_size=(160, 160))
    img = np.expand_dims(img,axis=0)
    embedding = get_embeddings(img)
    image = np.reshape(embedding, (1, 512))
    
    prediction = model(image, training=training_or_not)
    return [p for p in prediction], embedding


def get_image_prediction_training(image_embedding, model, training_or_not=True):
    image = np.reshape(image_embedding, (1, 512))
        
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    prediction = model(image, training=training_or_not)
    return [p for p in prediction]

def decode_image_prediction(predictions):
    decoded_predictions = []

    
    for i, probs in enumerate(predictions):
        if tf.is_tensor(probs):
            probs = probs.numpy()

        probs = np.squeeze(probs)

        if probs.ndim == 0 or (probs.ndim == 1 and probs.shape[0] == 1):
            predicted_class_index = (probs > 0.5).astype(int)
            decoded_label = LOADED_LABEL_ENCODERS[i].inverse_transform([predicted_class_index])[0]

        else:
            predicted_class_index = np.argmax(probs, axis=-1)
            decoded_label = LOADED_LABEL_ENCODERS[i].inverse_transform([predicted_class_index])[0]

        decoded_predictions.append(decoded_label)

    return np.array(decoded_predictions).reshape(1, -1)


def extract_face(frame, required_size=(160, 160)):
    
    if isinstance(frame, np.ndarray):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
    elif isinstance(frame, Image.Image):
        img = frame.convert('RGB')
    else:
        raise ValueError("Entrada inválida: deve ser numpy array (cv2) ou PIL.Image")

    boxes, probs = MTCNN.detect(img)

    if boxes is not None and len(boxes) > 0:
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = [int(coord) for coord in boxes[best_idx]]

        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img.crop((x1, y1, x2, y2))

        face_resized = face.resize(required_size)

        return np.asarray(face_resized)
    else:
        return None


def training_network_prediction(image_embedding, prediction_decoded, model, training_or_not):
    
    training_prediction = get_image_prediction_training(image_embedding, model, training_or_not)

    att_x_Prediction_ccd = {
        'Atributo': [],
        'Training prediction inferred class decoded': [],
        'Training prediction inferred class encoded': []
    }
    
    decoded_prediction = decode_image_prediction(training_prediction)

    for i in range(len(training_prediction)):
        probs = training_prediction[i]
        
        if tf.is_tensor(probs):
            probs = probs.numpy()

        probs = np.squeeze(probs)

        idx_target = prediction_decoded[i]

        if probs.ndim == 0 or probs.shape == ():
            prob_value = float(probs)
        elif probs.ndim == 1:
            prob_value = float(probs[idx_target])
        else:
            prob_value = float(probs.flatten()[idx_target])

        att_x_Prediction_ccd['Atributo'].append(ATRIBUTOS_CC[i])
        att_x_Prediction_ccd['Training prediction inferred class decoded'].append(decoded_prediction[0][i])
        att_x_Prediction_ccd['Training prediction inferred class encoded'].append(round(prob_value, 5))

    df_ = pd.DataFrame(att_x_Prediction_ccd)
    return df_, training_prediction


def inference_network_prediction(img, model, training_or_not):
    
    inferrence_pred, image_embedding = get_image_prediction(img, training_or_not, model=model)

    att_x_Prediction_ccd = {
        'Atributo': [],
        'Predicao decoded': [],
        'Predicao decoded index': [],
        'Predicao coded': []
    }
    
    decoded_prediction = decode_image_prediction(inferrence_pred)

    for i in range(len(inferrence_pred)):
        probs = inferrence_pred[i]

        if tf.is_tensor(probs):
            probs = probs.numpy()

        probs = np.squeeze(probs)

        if probs.ndim == 0 or probs.shape == ():  
            prob_value = float(probs)
            idx_max = 0 if prob_value <= 0.5 else 1

        elif probs.ndim == 1 and probs.shape[0] == 1:
            prob_value = float(probs[0])

        else:
            idx_max = np.argmax(probs)
            prob_value = float(probs[idx_max])

        att_x_Prediction_ccd['Atributo'].append(ATRIBUTOS_CC[i])
        att_x_Prediction_ccd['Predicao decoded'].append(decoded_prediction[0][i])
        att_x_Prediction_ccd['Predicao decoded index'].append(idx_max)
        att_x_Prediction_ccd['Predicao coded'].append(prob_value)

    df_ = pd.DataFrame(att_x_Prediction_ccd)
    return df_, image_embedding


def pred_relb_func(img_filepath, model, m=100):
    img = Image.open(img_filepath)
    df_pred_func, img_embedding = inference_network_prediction(img, model, False)
    
    inferred_class_prediction_index = df_pred_func["Predicao decoded index"].values
    
    df_relb = pd.DataFrame(columns=ATRIBUTOS_CC)

    df_train_it = pd.DataFrame(columns=["Atributo",
            "label da classe predita pelo modelo em INFERÊNCIA",
            "índice da classe predita pelo modelo em modo de INFERÊNCIA",
            "predição do modelo em TREINAMENTO para a classe predita pelo modelo em INFERÊNCIA",
            "label da classe predita pelo modelo em TREINAMENTO"])


    for i in range(m):
        
        df_inferred_class_training_prediction, subnet_prob_output = training_network_prediction(
            img_embedding, prediction_decoded=inferred_class_prediction_index, model=model, training_or_not=True
        )

        df_relb.loc[len(df_relb)] = df_inferred_class_training_prediction[
            "Training prediction inferred class encoded"
        ].values

        for j, att in enumerate(ATRIBUTOS_CC):

            inferred_class_index = inferred_class_prediction_index[j]
            inferred_class_label = LOADED_LABEL_ENCODERS[j].inverse_transform([inferred_class_index])[0]

            aux = subnet_prob_output[j]
            if tf.is_tensor(aux):
                aux = aux.numpy().squeeze()
            train_class_index = np.argmax(aux)

            train_class_pred_inferred_index = aux[inferred_class_index] if aux.ndim > 0 else aux

            train_class_label = LOADED_LABEL_ENCODERS[j].inverse_transform([train_class_index])[0]

            print(f"Atributo: {att}, na iteração {i}")
            print(f"= MODO DE INFERÊNCIA")
            print(f"  Índice da classe predita (idx={inferred_class_index})")
            print(f"  Label da classe predita (label={inferred_class_label})")
            print(f"= MODO DE TREINAMENTO")
            print(f"  Predição para a classe predita no modo de inferência: {train_class_pred_inferred_index:.5f}")
            print(f"  Label da classe predita (label={train_class_label})")
            print("-" * 80)
            
            df_train_it.loc[len(df_train_it)] = [att, inferred_class_label, inferred_class_index,
                                                 train_class_pred_inferred_index,
                                                 train_class_label]

    a_factor = 0.05
    att_reliability = {"Atributos": [], "Confiabilidade": []}

    for att in ATRIBUTOS_CC:
        one_class_predictions = df_relb[att]
        termo1 = one_class_media_ponderada(one_class_predictions, a_factor)
        termo2 = one_class_abs_diff_media(one_class_predictions, a_factor)

        result = termo1 - termo2
        att_reliability["Atributos"].append(att)
        att_reliability["Confiabilidade"].append(round(result, 5))

    df_pred_func["Confiabilidade"] = att_reliability["Confiabilidade"]
    df_pred_func = df_pred_func.drop(columns=["Predicao decoded index"])
    
    return df_pred_func, df_train_it


def compare_pred_func(models, image_path):
    
    pred_func_df = pd.DataFrame()
    training_it_df = pd.DataFrame()

    for i in range(len(models)):
        pred_func, training_it = pred_relb_func(img_filepath=image_path, model=models[i][0], m=m)
        
        pred_func["model"] = [models[i][1] for j in range(len(pred_func))]
        training_it["model"] = [models[i][1] for j in range(len(training_it))]

        pred_func_df = pd.concat([pred_func, pred_func_df], axis=0, ignore_index=True)
        training_it_df = pd.concat([training_it, training_it_df], axis=0, ignore_index=True)

    return pred_func_df, training_it_df

if __name__ == "__main__":
    
    m=100
    image_path = "../test_imgs/rafael_img2.png"
    
    models = [[MODEL_HYBRID, "mac_hybrid_200_2048_1e-05"]]

    df_pred, df_train = compare_pred_func(models, image_path)

    print(df_pred)#, df_train)
    
    df_pred.to_csv(f"mac_hybrid_{image_path.split('/')[-1]}.csv")
    df_train.to_csv(f"mac_hybrid_{image_path.split('/')[-1]}.csv")