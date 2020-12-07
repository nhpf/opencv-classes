import numpy as np                            # Facilita operações matemáticas
import cv2                                    # OpenCV
import json                                   # Biblioteca para ler os dados dos posters
from PIL import Image, ImageDraw, ImageFont   # Bilbioteca de manipulação de imagem


def load_coefficients(path):
    """ Lê coeficientes e matriz de correção a partir do arquivo guardado """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()  # Fechar o arquivo
    return [camera_matrix, dist_matrix]


def projection_matrix(calibration_matrix, homography_matrix):
    # Computar rotação e translação nos eixos x e y
    rot_and_transl = np.dot(np.linalg.inv(calibration_matrix), homography_matrix)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # Normalizar os vetores de rotação de translação
    l = np.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # Determinar a base ortonormal
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # Enfim, computar a matrix de projeção 3D a partir dos valores calculados
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(calibration_matrix, projection)


def prepare_title(im_width, title):
    # Fonte que será usada para desenhar o título no poster
    font_path = "..\\database\\Karla-Bold.ttf"
    font_size = 36
    font = ImageFont.truetype(font_path, font_size)

    # Diminui o tamanho da fonte até que o título caiba no poster
    while font.getsize(title)[0] > (im_width-10):
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)

    # Cria uma imagem preta com as dimensões to título
    im = Image.new('RGB', tuple(font.getsize(title)), (0, 0, 0))

    # Com a biblioteca PIL, define a ferramenta de desenho
    draw = ImageDraw.Draw(im)

    # Desenha o título no poster em verde
    draw.text((2, 0), title, (0, 255, 0), font=font)

    # Retorna a imagem no formato compatível com o OpenCV (numpy array)
    return np.array(im)[:, :, ::-1].copy()


def create_cube_points(shape=(20, 20), height=1):
    # Área do cubo deve ocupar 1/8 da área do poster
    poster_area = shape[0]*shape[1]
    side_length = np.sqrt(poster_area)/8

    # Para centralizar o cubo, calculamos onde o vértice superior deve estar
    cube_x = (shape[1] - side_length)/2
    cube_y = (shape[0] - side_length)/2

    # Altura da base de cada cubo para que esta fique distante de 1/4 de side_length
    # em relação à superfície do poster e à base dos outros cubos
    base = -0.25 * side_length * (5*height-4)

    return np.float32([ # Base
                        [cube_x, cube_y, base],
                        [cube_x, cube_y + side_length, base],
                        [cube_x + side_length, cube_y + side_length, base],
                        [cube_x + side_length, cube_y, base],
                        # Topo
                        [cube_x, cube_y, base - side_length],
                        [cube_x, cube_y + side_length, base - side_length],
                        [cube_x + side_length, cube_y + side_length, base - side_length],
                        [cube_x + side_length, cube_y, base - side_length]
                      ])


def draw_cube_in_img(img, imgpts):
    """ Desenha um cubo numa imagem a partir dos pontos dados """
    # Converte cada par de coordenadas [x,y] da imagem em inteiros.
    # A função reshape(-1,2) é usada para garantir que a variável imgpts seja uma matriz com dimensão X por 2 (x linhas, cada uma com 2 pontos)
    # Como x=8 vértices no cubo, o -1 podia ser substituído por 8 também
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Aqui conectamos os primeiros 4 pontos em vermelho, correspondentes à base do cubo
    # Os argumentos são imagem, vetor de pontos, -1 para desenhar todos os lados, cor e espessura 3.
    # Se um valor negativo for fornecido para a espessura, ele pinta a face inteira.
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -1)
    # O loop itera as variáveis [i de 0 a 3] e [j de 4 a 7]
    for i, j in zip(range(4), range(4, 8)):
        # Conectamos em verde o i-ésimo ponto de imgpts com o j-ésimo ponto de imgpts (conectar pontos da base com pontos do topo do cubo)
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 0, 255), 2)
    # Aqui conectamos os últimos 4 pontos em azul, correspondentes ao topo do cubo
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), -1)
    return img


def augment_posters(posters, calib_mtx, calib_dst):
    # Número mínimo de pontos que devem ser encontrados na câmera para determinar se o poster está ali ou não
    MIN_MATCHES = 48

    # Usar o algoritmo ORB (Oriented FAST and Rotated BRIEF) para detectar características-chave das imagens
    # nfeatures indica o número de pontos-chave da imagem do poster que serão usados para caracterizá-lo,
    # enquanto scoreType indica o critério de escolha dos pontos
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)

    # Para cada poster, cria imagens com o respectivo título
    title_images = []
    for p in posters:
        title_images.append(prepare_title(p['shape'][1], p['title']))

    # Webcam
    cap = cv2.VideoCapture(0)

    # A partir das dimensões do primeiro frame e da matriz de correção,
    # é gerada uma matriz para a câmera e a região de interesse (roi)
    _, img = cap.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calib_mtx, calib_dst, (w, h), 1, (w, h))

    # mapx, mapy são os fatores de correção que serão aplicados em cada frame para eliminar a distorção
    mapx, mapy = cv2.initUndistortRectifyMap(calib_mtx, calib_dst, None, newcameramtx, (w, h), 5)

    while True:
        # A cada iteração, captura um frame
        _, img2 = cap.read()

        # Retorna o frame corrigido a partir dos parâmetros 'mapx' e 'mapy'
        img2 = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

        # Limita a imagem apenas à região de interesse
        x, y, w, h = roi
        img2 = img2[y:y + h, x:x + w]

        # Usa o detector ORB para identificar os pontos-chave e descritores da imagem da webcam
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Para comparar as características-chave (vetores "descritors") do poster com as características da imagem da câmera,
        # Será usado um algoritmo FLANN (Fast approximate nearest neighbour) para essa comparação
        # index_params são parâmetros recomendados pela documentação do OpenCV
        index_params = dict(algorithm=6,            # Pontos serão associados com Locality Sensitivy Hashing
                            table_number=6,         # O número de tabelas de hashing
                            key_size=12,            # O tamanho da chave nas tabelas
                            multi_probe_level=2)    # O número de níves em que o algoritmo será executado
        # A função pede um argumento para parâmetros de pesquisa, que neste caso será deixado em branco
        search_params = {}
        # Configura o comparador FLANN com os parâmetros acima
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        posters_matches = []
        posters_good_matches = []
        for poster_index, poster in enumerate(posters):
            # Quando não é possível relacionar as imagens
            try:
                # Usa o comparador FLANN para relacionar pontos da câmera com pontos do poster
                posters_matches.append(flann.knnMatch(poster['descriptors'], des2, k=2))
                posters_good_matches.append([])

                # Filtra os pontos encontrados usando o teste de razão de Lowe
                posters_matches[poster_index] = [x for x in posters_matches[poster_index] if x and len(x) == 2]  # Garantir que só haverão pontos (x,y) no vetor
                for m, n in posters_matches[poster_index]:
                    if m.distance < 0.75 * n.distance:
                        posters_good_matches[poster_index].append(m)

            except Exception as e:
                print(e)

        # Seleciona o poster que tem um maior número de good_matches
        best_matches = max(posters_good_matches, key=lambda match: len(match))
        poster = posters[posters_good_matches.index(best_matches)]
        title_image = title_images[posters_good_matches.index(best_matches)]

        # Se o número de pontos encontrados for maior do que o mínimo arbitrado, consideramos que a imagem foi encontrada
        if len(best_matches) > MIN_MATCHES:
            # Bloco try para detectar erros na transformação de perspectiva
            try:
                # Extrair os pontos da câmera e da imagem identificados anteriormente
                src_pts = np.float32([poster['keypoints'][m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

                # Define a região da imagem identificada na câmera (homografia da imagem original)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Calcula a matriz de projeção levando em conta a matrix de calibração da câmera e a matriz de homografia
                transform_matrix = projection_matrix(calib_mtx, M)

                # Rotaciona a imagem com o título e combina com a imagem da câmera
                warped_title = cv2.warpPerspective(title_image, M, (img2.shape[1], img2.shape[0]))
                img2 = cv2.bitwise_or(warped_title, img2)

                # Cria um vetor com N cubos, onde N é o score do filme
                cube_array = [create_cube_points(shape=poster['shape'], height=x+1) for x in range(poster['rating'])]
                for cube in cube_array:
                    # Determina as coordenadas dos pontos do cubo através da matriz de projeção
                    imgpts = cv2.perspectiveTransform(cube.reshape(-1, 1, 3), transform_matrix)

                    # Substitui frame corrigido pelo frame corrigido com os cubos desenhados
                    img2 = draw_cube_in_img(img2, imgpts)

            # Se ocorrer, imprime o erro e apenas mostra a imagem da câmera
            except Exception as e:
                print(e)

        # Mostra a variável img2
        cv2.imshow('Reconhecido', img2)

        # Encerra o programa quando aperta a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # Carregar dados relativos aos posters, no arquivo db.json
    with open('../database/db.json', 'r') as j:
        db_file = json.loads(j.read())

    # Lê coeficientes da câmera e matriz de correção a partir do arquivo coeff.yml gerado na calibração
    mtx, dst = load_coefficients('../coeff.yml')

    # Transfere os dados dos posters para uma matriz e
    # adequa keypoints e descriptors de acordo com as classes requeridas pelo OpenCV
    posters = []
    for poster_data in db_file:
        poster = poster_data
        poster['keypoints'] = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2], _response=kp[3], _octave=kp[4], _class_id=kp[5]) for kp in poster_data['keypoints']]
        poster['descriptors'] = np.asarray(poster_data['descriptors'], dtype=np.uint8)
        posters.append(poster)

    # Mostra o rating e o título de cada poster detectado na webcam
    augment_posters(posters, mtx, dst)
