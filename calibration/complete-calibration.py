import numpy as np  # Facilita operações matemáticas
import cv2          # OpenCV
import glob         # Biblioteca para percorrer arquivos

# Critérios para calibrar. São feitas 30 iterações com precisão de subpixels 0.001 vezes o tamanho de um pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dir_imgs, prefix, image_format, square_size, width=9, height=6):
    """ Pega imagens do 'tabuleiro de xadrez' tiradas com uma câmera para gerar uma matriz de correção """
    # Prepara pontos  (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0) dependendo do tamanho do tabuleiro
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


def draw_cube_in_img(img, imgpts):
    """ Desenha um cubo numa imagem a partir dos pontos dados """
    # Converte cada par de coordenadas [x,y] da imagem em inteiros.
    # A função reshape(-1,2) é usada para garantir que a variável imgpts seja uma matriz com dimensão X por 2 (x linhas, cada uma com 2 pontos)
    # Como x=8 vértices no cubo, o -1 podia ser substituído por 8 também
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Aqui conectamos os primeiros 4 pontos em vermelho, correspondentes à base do cubo
    # Os argumentos são imagem, vetor de pontos, -1 para desenhar todos os lados, cor e espessura 3.
    # Se um valor negativo for fornecido para a espessura, ele pinta a face inteira.
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), 3)
    # O loop itera as variáveis [i de 0 a 3] e [j de 4 a 7]
    for i, j in zip(range(4), range(4, 8)):
        # Conectamos em verde o i-ésimo ponto de imgpts com o j-ésimo ponto de imgpts (conectar pontos da base com pontos do topo do cubo)
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)
    # Aqui conectamos os últimos 4 pontos em azul, correspondentes ao topo do cubo
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def draw_cube_from_camera(mtx, dist, width=9, height=6):
    """ Chama a função draw_cube_in_img para uma câmera calibrada """
    # Prepara pontos  (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0) dependendo do tamanho da câmera
    objp = np.zeros((height*width, 3), np.float32)              # Matriz com zeros para as interseções do tabuleiro. São 3 zeros para cada ponto (x, y, z)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)  # Preenche essa matriz de zeros com as coordenadas (x, y, 0) de acordo com as dimensões do tabuleiro

    # Se quiser gravar um vídeo, basta descomentar a linha debaixo e escrever out.write(frame) dentro do loop
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(cam.get(3)), int(cam.get(4))))

    # Define um cubo de aresta 3. A base do cubo tem coordenada z=0 e o topo do cubo tem coordenada z=-3 (o eixo "para fora" é negativo)
    cube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],       # Base
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])  # Topo

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

        # Converte a imagme para escala de cinza
        gray = cv2.cvtColor(corr, cv2.COLOR_BGR2GRAY)

        # Reconhece o tabuleiro
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # Caso encontrar o tabuleiro
        if ret:
            # Estima a orientação da câmera
            _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
            # Projeta os pontos na superfície do tabuleiro a partir da orientação determinada
            imgpts, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)
            # Substitui frame corrigido pelo frame corrigido com cubo
            corr = draw_cube_in_img(corr, imgpts)

        # Desenha o frame corrigido, tendo encontrado o tabuleiro ou não
        cv2.imshow('Alinhado', corr)

        # Desenha o frame original numa janela ao lado
        cv2.imshow('Original', frame)

        # Encerra o programa quando aperta a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # Calibrar com imagens output1.png, output2.png, ... Dentro do diretório '/images'
    ret, mtx, dist, rvecs, tvecs = calibrate('../images', 'output', 'png', 0.02, 9, 6)

    # Salvar coeficientes num arquivo ../coeff.yml
    save_coefficients(mtx, dist, '../coeff.yml')

    # Recuperar esses coeficientes do arquivo
    mtx, dist = load_coefficients('../coeff.yml')
    # Mostra, lado a lado, a imagem original e a imagem corrigida com um cubo
    draw_cube_from_camera(mtx, dist)
