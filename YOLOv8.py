import cv2
import numpy as np
from ultralytics import YOLO
import os


def load_model(model_path):

    """
    Esta función es la encargada de cargar el modelo YOLO entrenado
    desde una ruta especifica y retorna el modelo cargado.
    """

    model = YOLO(model_path)
    return model


def segment_and_mask_frame(model, frame, grada_class_id):

    """
    Esta función es la encargada de realizar la predicción usando el modelo YOLO, aplicar la segmentación por instancia en el frame,
    encontrar la clase "grada" y aplicar una máscara negra sobre las áreas identificadas. 
    Retorna el frame procesado con la máscara aplicada y la máscara binaria.
    """

    # Realizar la predicción
    results = model(frame)      

    # Obtener las máscaras y las etiquetas de las predicciones
    masks = results[0].masks.data.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy().astype(int)

    # Crear una máscara binaria inicial
    mask_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Iterar sobre las etiquetas y las máscaras para encontrar la clase "grada"
    for mask, label in zip(masks, labels):
        if label == grada_class_id:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_binary = np.maximum(mask_binary, (mask_resized > 0.5).astype(np.uint8) * 255)

    # Crear una imagen negra del mismo tamaño que la imagen original
    black_image = np.zeros_like(frame)

    # Aplicar la máscara binaria a la imagen negra
    black_masked = cv2.bitwise_and(black_image, black_image, mask=mask_binary)

    # Aplicar la máscara binaria a la imagen original
    frame_masked = cv2.bitwise_and(frame, frame, mask=255 - mask_binary)

    # Combinar la imagen original con la imagen negra en las áreas de la máscara
    masked_frame = cv2.add(frame_masked, black_masked)

    return masked_frame, mask_binary


def process_video(model, video_path, output_path, grada_class_id):

    """
    Esta función es la encargada de procesar un video completo utilizando el modelo YOLO. 
    Aplica la segmentación y la máscara negra en cada frame del video, guarda el video procesado y los frames individuales 
    y muestra una vista previa.
    No retorna ningún valor.
    """

    # Capturar el video
    cap = cv2.VideoCapture(video_path)

    # Obtener propiedades del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Definir el códec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Crear la carpeta de salida si no existe
    os.makedirs(output_frame_dir, exist_ok=True)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Realizar la segmentación y aplicar la máscara negra en las gradas
            masked_frame, mask = segment_and_mask_frame(model, frame, grada_class_id)

            # Escribir el frame procesado en el video de salida
            out.write(masked_frame)

            # Guardar el fotograma procesado como una imagen
            frame_filename = os.path.join(output_frame_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, masked_frame)
            frame_count += 1

            # Ajustar el tamaño del frame para vista previa
            preview_frame = cv2.resize(masked_frame, None, fx=0.5, fy=0.5)

            # Mostrar el frame procesado para vista previa
            cv2.imshow('Vista Previa', preview_frame)


            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"No se pudo aplicar la máscara en el frame: {e}")

    # Liberar los objetos de captura y escritura de video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Ruta del modelo entrenado y del video de prueba
model_path = r'C:\Users\PC\Documents\Proyectos\Challenge\modelos\model3\best.pt'  
video_path = r'C:\Users\PC\Documents\Proyectos\Challenge\Muestras\videomuestras\video6.mp4'  
output_path = r'C:\Users\PC\Documents\Proyectos\Challenge\Muestras\video_output.mp4'  
output_frame_dir = r'C:\Users\PC\Documents\Proyectos\Challenge\Muestras\frames'

# ID de la clase "grada"
grada_class_id = 2

# Cargar el modelo entrenado
model = load_model(model_path)

# Procesar el video
process_video(model, video_path, output_path, grada_class_id)