import numpy as np     # Facilita operações matemáticas
import cv2             # OpenCV
import importlib.util  # Bilbioteca para importar outros arquivos .py

# Importa as funções definidas no arquivo complete-calibration.py
spec = importlib.util.spec_from_file_location('complete-calibration', '../calibration/complete-calibration.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def keypoints_debug(img):
    """ Função para testar a funcionalidade do algoritmo SURF """
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SURF', kp_img)


def detect_image_surf(img1, mtx, dist):
    # Número mínimo de pontos-chave criados na imagem do poster para caracterizá-lo
    minHessian = 500

    # Número mínimo de pontos que devem ser encontrados na câmera para determinar se o poster está ali ou não
    minMatches = 80

    # Usar o algoritmo SURF (Speeded-Up Robust Features) para detectar características-chave das imagens
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)

    # Usa o detector SURF para identificar os pontos-chave e descritores do poster
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)

    # Para comparar as características-chave (vetores "descritors") do poster com as características da imagem da câmera,
    # Será usado um algoritmo FLANN (Fast approximate nearest neighbour) para essa comparação
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

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
        _, img2 = cap.read()

        # Retorna o frame corrigido a partir dos parâmetros 'mapx' e 'mapy'
        img2 = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

        # Limita a imagem apenas à região de interesse
        x, y, w, h = roi
        img2 = img2[y:y + h, x:x + w]

        # Usa o detector SURF para identificar os pontos-chave e descritores da imagem da webcam
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

        # Usa o comparador FLANN para relacionar pontos da câmera com pontos do poster
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

        # Filtra os pontos encontrados usando o teste de razão de Lowe
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Se o número de pontos encontrados for maior do que o mínimo arbitrado, consideramos que a imagem foi encontrada
        if len(good_matches) > minMatches:
            # Bloco try para detectar erros na transformação de perspectiva
            try:
                # Extrair os pontos da câmera e da imagem identificados anteriormente
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Define a região da imagem identificada na câmera (homografia da imagem original)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # Com as dimensões da imagem original, define uma matriz de pontos que constituem o contorno dela
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # Converte a matriz do contorno da imagem para a perspectiva da câmera
                dst = cv2.perspectiveTransform(pts, M)
                # Desenha uma linha em volta da imagem identificada na webcam
                img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Se ocorrer, imprime o erro e apenas mostra a imagem da câmera
            except Exception as e:
                print(e)

        # Mostra a variável img2
        cv2.imshow('Reconhecido', img2)

        # Encerra o programa quando aperta a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_image_orb(img1, mtx, dist):
    # Número mínimo de pontos que devem ser encontrados na câmera para determinar se o poster está ali ou não
    MIN_MATCHES = 32

    # Usar o algoritmo ORB (Oriented FAST and Rotated BRIEF) para detectar características-chave das imagens
    # nfeatures indica o número de pontos-chave da imagem do poster que serão usados para caracterizá-lo,
    # enquanto scoreType indica o critério de escolha dos pontos
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)

    # Usa o detector ORB para identificar os pontos-chave e descritores do poster
    kp1, des1 = orb.detectAndCompute(img1, None)

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
        _, img2 = cap.read()

        # Retorna o frame corrigido a partir dos parâmetros 'mapx' e 'mapy'
        corr = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

        # Limita a imagem apenas à região de interesse
        x, y, w, h = roi
        img2 = corr[y:y + h, x:x + w]

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

        # Usa o comparador FLANN para relacionar pontos da câmera com pontos do poster
        matches = flann.knnMatch(des1, des2, k=2)  # Error on blurred

        # Filtra os pontos encontrados usando o teste de razão de Lowe
        good_matches = []
        matches = [x for x in matches if x and len(x) == 2]  # Garantir que só haverão pontos (x,y) no vetor
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Se o número de pontos encontrados for maior do que o mínimo arbitrado, consideramos que a imagem foi encontrada
        if len(good_matches) > MIN_MATCHES:
            # Bloco try para detectar erros na transformação de perspectiva
            try:
                # Extrair os pontos da câmera e da imagem identificados anteriormente
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Define a região da imagem identificada na câmera (homografia da imagem original)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # Com as dimensões da imagem original, define uma matriz de pontos que constituem o contorno dela
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # Converte a matriz do contorno da imagem para a perspectiva da câmera
                dst = cv2.perspectiveTransform(pts, M)

                # Desenha uma linha em volta da imagem identificada na webcam
                img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Se ocorrer, imprime o erro e apenas mostra a imagem da câmera
            except Exception as e:
                print(e)

        # Mostra a variável img2
        cv2.imshow('Reconhecido', img2)

        # Encerra o programa quando aperta a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    im1 = cv2.imread('../images/calculo.png', 0)  # Lê a imagem do poster em escala de cinza
    mtx, dist = calibration.load_coefficients('../coeff.yml')
    detect_image_orb(im1, mtx, dist)
