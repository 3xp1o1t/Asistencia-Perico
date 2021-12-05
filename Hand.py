import math
import time
import cv2
import mediapipe as mp

# --- Clase con funciones relacionadas a la deteccion de manos y dedos --- #
# --- Metodos tomados de Aprende e Ingenia --- #
# --- La mayoria de los metodos los modifique y pase a ingles en caso de obtener ayuda externa --- #

class Hand():

    # Constructor parametros de inicio #
    def __init__(self, mode = False, max_hands = 1, detection_confidence = 0.5, tracking_confidence = 0.5):
        """ Constructor, inicializa la configuracion para detectar y trackear las manos."""
        # Static = False (Video/Stream), True only for image
        self.mode = mode                    
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        # Constructor para detectar manos.
        self.hands = self.mp_hands.Hands(static_image_mode = self.mode, max_num_hands = self.max_hands, min_detection_confidence = self.detection_confidence, min_tracking_confidence = self.tracking_confidence)  
        # Objecto para dibujar
        self.draw = mp.solutions.drawing_utils      
        # Estilo para los puntos y conexiones 
        self.style_points = self.draw.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 3)
        self.style_connections = self.draw.DrawingSpec(color = (10, 255, 10), thickness = 2)
        # https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
        self.finger_index = [4, 8, 12, 16, 20]      
        

    # Metodo para localizar las manos
    def find_hand(self, frame, draw_landmarks = True):
        """
        find_hand: Localiza y dibuja la mano detectada
        :param frame: Fotograma a analizar
        :param want_draw_landmarks: True si quiere dibujar las marcas de mano, False caso contrario.
        :return frame: Fotograma procesado
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inicia la deteccion de manos
        self.results = self.hands.process(frame_rgb)

        # Aqui hay 2 formas, la de Aprende e Ingenia y la de la docu, usando with, pero requerimos retornar el fotograma
        # con la deteccion para pintarlo.
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                # Si want_draw_lendmarks = True, entonces dibujamos. si no salimos y retornamos el frame.
                if draw_landmarks:
                    self.draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS, self.style_points, self.style_connections)

        return frame

    # Metodo para localizar y dibujar los puntos y conexiones de la mano 
    def find_pos(self, frame, hand_num = 0, draw_landmarks = True):
        """
        find_pos: Dado un frame, buscar y pintar marcas y conexiones
        :param frame: Fotograma donde dibujara.
        :param hand_num: El numero de mano dado un index, 0, 1
        :param draw_landmarks: True si quiere dibujar las manos, False no.
        :return points: Cordenadas de cada punto(21) con sus posiciones en X y Y en pixeles.
        """
        points_x = []
        points_y = []
        self.points = []

        if self.results.multi_hand_landmarks:
            # Mano actual
            hand = self.results.multi_hand_landmarks[hand_num]
            for (index, point) in enumerate(hand.landmark):
                # Escala
                height, width, _ = frame.shape
                # Pixeles del fotograma actual
                x, y = int(point.x * width), int(point.y * height)
                # Almacenar los puntos encontrados
                points_x.append(x)
                points_y.append(y)

                # Puntos para formar las conexiones
                self.points.append([point, x, y])

                if draw_landmarks:
                    # Dibujar el circulo en los puntos encontrados.
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), cv2.FILLED)
            
        return self.points

    # Metodo para detectar el dedo que este arriba
    def is_finger_up(self):
        fingers = []
        # Si no es un indice de dedo valido
        if self.points[self.finger_index[0]][1] > self.points[self.finger_index[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for index in range(1, 5):
            if self.points[self.finger_index[index]][2] < self.points[self.finger_index[index]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    # Metodo para medir la distancia entre los dedos.
    def distance(self, point_1, point_2, frame, draw_landmarks = True, radius = 10, thickness = 3):
        """
        distance: Mide la distancia entre el punto 1 y el punto 2
        :param point_1: Punto inicial
        :param point_2: Punto final
        :param frame: Fotograma actual
        :param draw_landmarks: Si quiere dibujar el circulo y la conexion entre los 2 puntos
        :param radius: Radio del circulo para dibujar en el punto 1 y 2
        :param thickness: Grosor de la linea a dibujar como conexion.
        :return: Distancia euclidiana, fotograma dibujado y lista con los puntos
        """
        x1, y1 = self.points[point_1][1::]
        x2, y2 = self.points[point_2][1::]
        x, y = (x1 + x2) // 2, (y1 + y2) // 2

        if draw_landmarks:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            cv2.circle(frame, (x1, y1), radius, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), radius, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x, y), radius, (0, 255, 0), cv2.FILLED)
        
        # Calcular la distancia euclidiana
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, x, y]

def main():
    ptime = 0
    ctime = 0

    # Obtenemos la camara por default
    cap = cv2.VideoCapture(0)

    hand_detector = Hand()

    while True:
        ret, frame = cap.read()
        # Enviar la imagen a find_hand
        frame = hand_detector.find_hand(frame)
        points = hand_detector.find_pos(frame)

        if len(points) != 0:
            print(points[4])

        cv2.imshow("Mano", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()