import numpy as np  # Facilita operações matemáticas
import cv2          # OpenCV
import glob         # Biblioteca para percorrer arquivos
import json         # Biblioteca para salvar os dados dos posters

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


def generate_keypoints(poster_src):
    # Lê a imagem cujo endereço é poster_src
    poster_img = cv2.imread(poster_src, 0)

    # Usar o algoritmo ORB (Oriented FAST and Rotated BRIEF) para detectar características-chave das imagens
    # nfeatures indica o número de pontos-chave da imagem do poster que serão usados para caracterizá-lo,
    # enquanto scoreType indica o critério de escolha dos pontos
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)

    # Usa o detector ORB para identificar os pontos-chave e descritores do poster
    keypoints, descriptors = orb.detectAndCompute(poster_img, None)

    # Converte um vetor de cv2.KeyPoint para um vetor com as estruturas nativas do python
    kps = []
    for kp in keypoints:
        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        kps.append(temp)

    # Retorna os keypoints e descriptors
    return [kps, descriptors, poster_img.shape]


if __name__ == '__main__':
    print("Começando a calibração...")

    # Calibrar com imagens output1.png, output2.png, ... Dentro do diretório '/images'
    ret, mtx, dist, rvecs, tvecs = calibrate('../images', 'output', 'png', 0.02, 9, 6)

    print("Câmera calibrada!\n")

    # Salvar coeficientes num arquivo ../coeff.yml
    save_coefficients(mtx, dist, '../coeff.yml')

    print("Começando a carregar os posters...")

    # Dados de cada poster (hard-coded)
    posters = [['../images/sharknado.png', 'Sharknado', 2], ['../images/forceawakens.jpg', 'Star Wars the Force Awakens', 4]]

    # Prepara o vetor de dicionários que será colocado no arquivo JSON
    data = []
    for poster in posters:
        # Gera os pontos-chave e descritores para cada poster
        keypoints, descriptors, shape = generate_keypoints(poster[0])

        # Cria um dicionário com os dados do poster
        poster_dict = {
            'src': poster[0],
            'title': poster[1],
            'rating': poster[2],
            'shape': shape,
            'keypoints': keypoints,
            'descriptors': descriptors.tolist(),
        }

        # Acrescenta esse dicionário ao vetor "data"
        data.append(poster_dict)

        print(f"\tPoster {poster[1]} concluído!")

    # Salva o vetor "data" no arquivo db.json
    with open('../database/db.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("\nPrograma concluído!")
