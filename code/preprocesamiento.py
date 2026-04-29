import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Configuracion de MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False)
hands = mp_hands.Hands(static_image_mode=False)

# Configuracion de directorios
input_dir = "Dataset"  # Carpeta base con las clases
output_file = "gesture_dataset12clases_5aum.npy"  # Archivo para guardar el dataset

# Clases y etiquetas
classes = {"adios": 0, "como_estas": 1, "gracias": 2, "hola": 3, "papa": 4, "por_favor": 5,
           "agua": 6, "caminar": 7, "comer": 8, "jugar": 9, "familia": 10, "mama": 11}

data = []  # Para almacenar las secuencias de puntos clave
labels = []  # Para almacenar las etiquetas

# Funcion para extraer puntos clave de un frame
def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar puntos clave
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    # Extraer coordenadas de pose
    keypoints = []
    if pose_results.pose_landmarks:
        for idx in [11, 12, 13, 14, 15, 16]:  # Puntos relevantes del cuerpo
            lm = pose_results.pose_landmarks.landmark[idx]
            keypoints.append(lm.x)
            keypoints.append(lm.y)
    else:
        keypoints.extend([0.0, 0.0] * 6)  # Rellenar con ceros

    # Extraer coordenadas de manos
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
    else:
        keypoints.extend([0.0, 0.0] * 21)  # Rellenar con ceros

    return np.array(keypoints)

# Ajustar la cantidad de frames a 60
def adjust_frames(video_data, target_frames=60):
    current_frames = len(video_data)
    if current_frames == target_frames:
        return video_data  # No necesita ajustes
    elif current_frames < target_frames:
        # Interpolar frames para alcanzar 60
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return [video_data[i] for i in indices]
    else:
        # Recortar frames al inicio o al final
        return video_data[:target_frames]

# Funcion para aplicar aumentos de datos
def augment_data(video_data):
    augmented_videos = []

    for video in video_data:
        # Flip horizontal
        flipped = [np.flip(frame, axis=0) for frame in video]
        augmented_videos.append(flipped)

        # Anadir rotacion y desplazamiento
        for frame in video:
            # Rotacion y desplazamiento en ejes
            rotated_frame = frame + np.random.uniform(-0.1, 0.1, size=frame.shape)
            augmented_videos.append(rotated_frame)

    return augmented_videos

# Procesar los videos en cada carpeta
for class_name, label in classes.items():
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"Procesando clase '{class_name}'...")
    all_video_data = []  # Para almacenar los videos procesados de esta clase

    for video_name in tqdm(os.listdir(class_path), desc=f"Procesando clase '{class_name}'"):
        video_path = os.path.join(class_path, video_name)
        cap = cv2.VideoCapture(video_path)

        video_data = []  # Para almacenar los frames del video actual

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = extract_keypoints(frame)
            if len(keypoints) == 96:  # Asegurar consistencia en los datos
                video_data.append(keypoints)

        cap.release()

        # Ajustar la cantidad de frames a 60
        if video_data:
            adjusted_video = adjust_frames(video_data, target_frames=60)
            all_video_data.append(adjusted_video)
            data.append(adjusted_video)
            labels.append(label)

    # Aumentar datos para esta clase
    if all_video_data:
        print(f"Aumentando datos para la clase '{class_name}'...")
        augmented_videos = augment_data(all_video_data)

        for augmented_video in augmented_videos:
            data.append(augmented_video)
            labels.append(label)

# Convertir a formato NumPy y guardar
data = np.array(data)
labels = np.array(labels)

np.save(output_file, {"data": data, "labels": labels})
print(f"Dataset guardado en {output_file}.")
print(f"Forma del dataset: {data.shape}, Etiquetas: {labels.shape}")
