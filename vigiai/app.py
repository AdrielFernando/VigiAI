import cv2
import numpy as np


# Função para redimensionar o vídeo
def resize_video(frame, width=10, height=20, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(frame, dim, interpolation=inter)
    return resized

# Função para detectar pessoas
def detect_person(frame):
    hog = cv2.HOGDescriptor() #incializa o objeto HOG 

    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # nesse codigo o metodo HOGDescriptor vai trazer pessos prdefinidos pra detecção de pessoas e o set configura o hog pra usar esses pesos 
    
    rects, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05) 
    
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame, rects

# Função para verificar se a pessoa atravessou a linha vertical
def check_crossed_line(current_rects, line_positionX):
    crossed = False
    for (x, y, w, h) in current_rects:
        right_x = x + w

        if x < line_positionX < right_x:
            crossed = True
            break

    return crossed

# Função principal
def main():

    video_capture = cv2.VideoCapture('testoficial.mp4') # faz a leitura do video

    line_positionX = 100  # cordenada do ponto X da line
    line_positionY = 0  # cordenada do ponto Y da line


    while True:

        ret, frame = video_capture.read() #enquanto o ret retornar true a leitura foi bem sucedida e o frame continua sendo atribuido 
        if not ret:
            break
        
        frame = resize_video(frame, width=640) #passa o frema e a altura pra ajustar a dimensão de forma proporcional 

        frame = cv2.flip(frame, 1) #espelha o frame horizontalmente 

        frame, current_rects = detect_person(frame)  #função que retorna dois valores - o novo frame após a detctar as pessoa e faze o bouning box e os valores da cordenada do própio bounding box 

        cv2.line(frame, (line_positionX, line_positionY), (line_positionX, frame.shape[0]), (0, 0, 255), 2) # aqui vai ser passado o frame, as coodernada e style da Line que vai ser traçado no propio frame
        
        crossed = check_crossed_line(current_rects, line_positionX) #a parte onde faço a comparação da coodereada do bounding box e da linha que delimitei pra definir se alguém está ultrapassando a linha 
        if crossed: 
            print("Alguem atravessou a linha!")
        
        cv2.imshow('Video', frame) #exibo o frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
