import argparse
import importlib.util
import os
import json
import cv2
import numpy as np

# Importa as funções definidas no arquivo calibration.py
spec = importlib.util.spec_from_file_location('calibration', './calibration/calibration.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)

# Importa as funções definidas no arquivo detection.py
spec = importlib.util.spec_from_file_location('detection', './detection/detection.py')
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)


def main_calibration(calibration_args, update_args):
    if calibration_args:
        print("Começando a calibração...")
        img_path, img_prefix, img_format = calibration_args

        # Calibrar com imagens que obedecem a regex [img_path]/[img_prefix].*\.[img_format]
        _, mtx, dist, _, _ = calibration.calibrate(img_path, img_prefix, img_format, 0.02, 9, 6)

        print("Câmera calibrada!\n")

        # Salvar coeficientes num arquivo ./coeff.yml
        calibration.save_coefficients(mtx, dist, './coeff.yml')

    if update_args:
        print("Começando a carregar os posters...")
        movies_path = update_args

        # Extrai os dados de cada poster a partir de um arquivo texto
        with open(movies_path, "r") as movies_file:
            movies = movies_file.readlines()
            posters = [[attribute.strip() for attribute in movie.split(',')] for movie in movies]

        # Prepara o vetor de dicionários que será colocado no arquivo JSON
        data = []
        for poster in posters:
            # Se algum arquivo de poster não existir, sai do programa
            if not os.path.isfile(poster[0]):
                print(f"Arquivo não encontrado: {poster[0]}")
                exit()

            # Gera os pontos-chave e descritores para cada poster
            keypoints, descriptors, shape = calibration.generate_keypoints(poster[0])

            # Cria um dicionário com os dados do poster
            poster_dict = {
                'src': poster[0],
                'title': poster[1],
                'rating': int(poster[2]),
                'shape': shape,
                'keypoints': keypoints,
                'descriptors': descriptors.tolist(),
            }

            # Acrescenta esse dicionário ao vetor "data"
            data.append(poster_dict)

            print(f"\tPoster {poster[1]} concluído!")

        # Salva o vetor "data" no arquivo db.json
        with open('./database/db.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    if calibration_args or update_args:
        print("\nCalibração concluída!")


def main_augmentation():
    # Carregar dados relativos aos posters, no arquivo db.json
    with open('./database/db.json', 'r') as j:
        db_file = json.loads(j.read())

    # Lê coeficientes da câmera e matriz de correção a partir do arquivo coeff.yml gerado na calibração
    mtx, dst = detection.load_coefficients('./coeff.yml')

    # Transfere os dados dos posters para uma matriz e
    # adequa keypoints e descriptors de acordo com as classes requeridas pelo OpenCV
    posters = []
    for poster_data in db_file:
        poster = poster_data
        poster['keypoints'] = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2], _response=kp[3], _octave=kp[4], _class_id=kp[5]) for kp in poster_data['keypoints']]
        poster['descriptors'] = np.asarray(poster_data['descriptors'], dtype=np.uint8)
        posters.append(poster)

    # Mostra o rating e o título de cada poster detectado na webcam
    detection.augment_posters(posters, mtx, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Augmented Posters!\n'
                                                 'Este programa detecta posters a partir da webcam do usuário e nelas mostra o título do filme '
                                                 'e um conjunto de cubos representando o score do filme (1 a 5).\n\n'
                                                 'Para calibrar a câmera, é necessário imprimir um tabuleiro com 10x7 quadrados de 2cm '
                                                 'de lado e tirar cerca de 20 fotos, organizadas num mesmo diretório e com prefixo e formato comuns.\n\n'
                                                 'Para atualizar o banco de dados, basta inserir o caminho para um arquivo .txt contendo, em cada linha '
                                                 'o caminho do arquivo de imagem com o poster, o título do filme e o score separados por vírgula.\n')
    parser.add_argument_group('--calibrate')
    parser.add_argument("--calibrate", nargs=3, metavar=('[img_path]', '[img_prefix]', '[img_format]'),
                        help='Calibrar a câmera a partir de fotos tiradas com ela.\n'
                             'As imagens de calibração obedecem a regex [img_path]/[img_prefix].*\\.[img_format]\n\n'
                             'Exemplo de uso: app.py --calibrate images output png')
    parser.add_argument("--update-db", metavar='[txt_file]',
                        help='Atualizar o banco de dados a partir de um arquivo .txt\n'
                             'Exemplo de arquivo movies.txt:\n\t./images/filme1.jpg, Sharknado, 2<\n\t./images/filme2.png, Star Wars, 4\n\n'
                             'Exemplo de uso: app.py --update movies.txt')
    args = parser.parse_args()

    main_calibration(args.calibrate, args.update_db)
    main_augmentation()
