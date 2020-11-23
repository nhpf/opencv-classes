import numpy as np  # Facilita operações matemáticas
import cv2          # OpenCV
import glob         # Biblioteca para percorrer arquivos

# Critérios para calibrar. São feitas 30 iterações com precisão de subpixels 0.001 vezes o tamanho de um pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dir_imgs, prefix, image_format, square_size, width=9, height=6):
    """ Pega imagens do 'tabuleiro de xadrez' tiradas com uma câmera para gerar uma matriz de correção """
    # Prepara pontos  (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0) dependendo do tamanho da câmera
    objp = np.zeros((height*width, 3), np.float32)              # Matriz com zeros para as interseções do tabuleiro. São 3 zeros para cada ponto (x, y, z)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)  # Preenche essa matriz de zeros com as coordenadas (x, y, 0) de acordo com as dimensões do tabuleiro

    # Transformar esse tabuleiro para escala real em metros
    objp = objp * square_size

    # Vetores para guardar os pontos nas imagens capturadads
    objpoints = []  # Pontos 3d da imagem
    imgpoints = []  # Pontos 2d no plano do tabuleiro

    # Pega o diretório especificado
    if dir_imgs[-1:] == '/':        # Se diretório terminar com /
        dir_imgs = dir_imgs[:-1]    # Retira / do nome do diretório
    images = glob.glob(dir_imgs+'/' + prefix + '*.' + image_format)  # Retorna todos os arquivos que tem prefixo 'prefix' e formato 'image_format'

    # Itera sobre as imagens de calibração
    for fname in images:
        # Lê cada imagem e transforma em escala de cinza
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontra os cantos do tabuleiro
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # Se encontrou
        if ret:
            # Adiciona os pontos 3d da imagem ao vetor objpoints
            objpoints.append(objp)

            # Ajusta os cantos detectados no tabuleiro de acordo com os critérios de precisão
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Adiciona os cantos ajustados ao vetor imgpoints
            imgpoints.append(corners2)

            # Esta função desenha os cantos do tabuleiro - usei apenas para debugging
            # img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    # Usa a função calibrateCamera para gerar a matriz de correção
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    """ Guarda a matriz de correção e coeficientes de distorção no arquivo determinado. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # Fechar o arquivo
    cv_file.release()


def load_coefficients(path):
    """ Lê coeficientes e matriz de correção a partir do arquivo guardado """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()  # Fechar o arquivo
    return [camera_matrix, dist_matrix]


if __name__ == '__main__':
    # Calibrar com imagens output1.png, output2.png, ... Dentro do diretório '/images'
    ret, mtx, dist, rvecs, tvecs = calibrate('../images', 'output', 'png', 0.02, 9, 6)

    # Salvar coeficientes num arquivo ../coeff.yml
    save_coefficients(mtx, dist, '../coeff.yml')

    # Recuperar esses coeficientes do arquivo
    mtx, dist = load_coefficients('../coeff.yml')

    # Webcam
    cap = cv2.VideoCapture(0)

    # A partir das dimensões do primeiro frame e da matriz de correção, é gerada uma matriz para a câmera e a região de interesse (roi)
    _, img = cap.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # mapx, mapy são os fatores de correção que serão aplicados em cada frame para eliminar a distorção
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    while True:
        # A cada iteração, captura um frame
        _, frame = cap.read()

        # Retorna o frame corrigido a partir dos parâmetros 'mapx' e 'mapy'
        corr = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Limita a imagem apenas à região de interesse
        x, y, w, h = roi
        corr = corr[y:y + h, x:x + w]

        # Mostra a imagem original
        cv2.imshow('Original', frame)

        # Mostra a imagem corrigida
        cv2.imshow('Corrigido', corr)

        # Encerra o programa quando aperta a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
