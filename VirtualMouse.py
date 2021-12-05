import cv2
import numpy as np
from Hand import Hand
import pyautogui

cam_width, cam_height = 640, 480
# Rango donde se puede interactuar con el mouse
interactive_box = 100
# ancho, alto pantalla
screen_width, screen_height = pyautogui.size()
# Suavizado para que no parpadee mucho el mouse
smoothed = 5
# Recibiran la informacion de la ubicacion de la mano.
# Previous ubicacion, current ubicacion.
pubix, pubiy = 0, 0
cubix, cubiy = 0, 0

pyautogui.FAILSAFE = False

# Iniciar la camara, con un ancho y alto por defecto
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

# Clase para detectar una sola mano y sus conexiones.
hand_detector = Hand()

# Iniciamos el reconocimiento de las manos
while True:
    ret, frame = cap.read()
    # Enviar la imagen a find_hand
    frame = hand_detector.find_hand(frame)
    points = hand_detector.find_pos(frame)

    if len(points) != 0:
        x1, y1 = points[8][1:]              # 8 = Dedo indice
        x2, y2 = points[12][1:]             # 12 = Dedo medio

        print('Pos Indice: ', x1, y1, '\nPos Medio: ', x2, y2)

        # Comprobar que dedos estan levantados.
        fingers = hand_detector.is_finger_up()      

        # Cuadro donde podra mover el mouse.
        cv2.rectangle(frame, (interactive_box, interactive_box), (cam_width - interactive_box, cam_height - interactive_box), (0, 0, 0), 2)

        # Si el dedo indice esta arriba, pero el medio abajo
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (interactive_box, cam_width - interactive_box), (0, screen_width))
            y3 = np.interp(y1, (interactive_box, cam_height - interactive_box), (0, screen_height))

            # Suavizar los valores para que no parpadie mucho el mouse
            cubix = pubix + (x3 - pubix) / smoothed
            cubiy = pubiy + (y3 - pubiy) / smoothed

            # Mover el mouse a la posicion del dedo indice.
            pyautogui.moveTo(screen_width - cubix, cubiy)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 0), cv2.FILLED)
            pubix, pubiy = cubix, cubiy
        
        # Si los 2 dedos estan arriba = Modo click
        # Dedo indice (indicador), dedo de en medio Click
        if fingers[1] == 1 and fingers[2] == 1:
            # Encontrar la distancia entre los dedos
            length, frame, line = hand_detector.distance(8, 12, frame)
            print("Distancia entre dedos: ", length)

            if length < 30:
                cv2.circle(frame, (line[4], line[5]), 10, (0, 255, 0), cv2.FILLED)

                # Mandamos click
                pyautogui.click()

    cv2.imshow("Mouse Virtual", frame)
    k = cv2.waitKey(1)

    if k == 27:
        break

#salimos
cap.release()
cv2.destroyAllWindows()
