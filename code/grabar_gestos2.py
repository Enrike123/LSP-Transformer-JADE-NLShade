import cv2
import time
import pygame

# Inicializar pygame para manejar sonido
pygame.mixer.init()

# Configuración inicial
gesture_name = "ayer"  # Nombre base del archivo (cambia según el gesto)
video_count = 100  # Cantidad de videos a grabar
frame_width = 640  # Ancho del video
frame_height = 480  # Altura del video
fps = 30  # Cuadros por segundo
video_duration = 2  # Duración de cada video en segundos
alert_sound = "alert.mp3"  # Archivo de sonido de aviso
pause_between_videos = 1  # Pausa entre grabaciones (en segundos)

# Iniciar captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

# Función para reproducir el sonido de aviso
def play_sound():
    pygame.mixer.music.load(alert_sound)
    pygame.mixer.music.play()

# Cuenta regresiva inicial para posicionarse
print("Preparándose para grabar...")
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret:
        print("Error:_No_se_pudo_capturar el cuadro.")
        break
    # Mostrar contador en pantalla
    cv2.putText(frame, f"Comenzando en {i}...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('Preparándose...', frame)
    cv2.waitKey(1000)  # Esperar 1 segundo

# Comienza el proceso de grabación
print(f"Preparándose para grabar {video_count} videos de gestos '{gesture_name}'.")

for i in range(100, video_count + 1):
    output_filename = f"{gesture_name}{i}.mp4"  # Generar nombre del archivo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Aviso sonoro
    print(f"Preparando para grabar video {i}...")
    play_sound()  # Reproduce el sonido de aviso
    time.sleep(1)  # Espera 1 segundo para que el usuario prepare el gesto

    # Graba el video de 2.5 segundos
    print(f"Grabando video {i}: {output_filename}")
    frame_count = 0
    max_frames = int(fps * video_duration)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break

        # Mostrar "Grabando..." en pantalla
        #cv2.putText(frame, "Grabando...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Grabando...', frame)

        # Guardar el cuadro
        out.write(frame)
        frame_count += 1

        # Salir si presionas la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Grabacion interrumpida.")
            break

    out.release()  # Liberar el archivo de video
    print(f"Video {i} guardado como {output_filename}.")

    # Pausa entre grabaciones
    time.sleep(pause_between_videos)

print("Proceso de grabación completado.")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
