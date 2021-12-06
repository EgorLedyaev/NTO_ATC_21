import helpers
import cv2
import random
import numpy as np
import os
import eval
import tensorflow as tf

def load_data():
    """ Формирование списка с изображениями и метками к ним
    """

    TEST_IMAGE_LIST = helpers.load_dataset("test_images/")
    random.shuffle(TEST_IMAGE_LIST)
    return TEST_IMAGE_LIST


# Получение списка неклассифицированных изображений
def get_misclassified_images(test_images, model, standart_signs_list):
    """Определение точности работы алгоритма
    Сравниваются результаты классификации вашего алгоритма и истинныме метки

    Входные данные: массив с тестовыми изображениями и метками к ним
    Выходные данные: массив с неправильно классифицированными метками

    Этот код используется для тестирования и не должен изменяться
    """
    misclassified_images_labels = []

    for image in test_images:
        # получение изображения и метки
        im = image[0]
        true_label = image[1]
        # метки должны быть в виде списка
        #print(true_label)
        assert (len(true_label) == 8), "Метка имеет не верную длинну (8 значений)"

        # Получение метки из написанного Вами классификатора
        predicted_label = eval.predict_label(im, model, standart_signs_list)
        assert (len(predicted_label) == 8), "Метка имеет не верную длинну (8 значений)"

        # Сравнение реальной и предсказанной метки
        if (predicted_label != true_label):
            # Если значения меток не совпадают, то изображение помечается как неклассифицированное
            misclassified_images_labels.append((im, predicted_label, true_label))
            print(predicted_label, true_label)

    # Возвращение неклассифицированных изображений [image, predicted_label, true_label]
    return misclassified_images_labels


def main():

    model = eval.load_final_model()
    standart_signs_list = eval.get_standatr_signs()
    TEST_IMAGE_LIST = load_data()
    MISCLASSIFIED = get_misclassified_images(TEST_IMAGE_LIST, model, standart_signs_list)

    # вычисление точности
    total = len(TEST_IMAGE_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Точность: ' + str(accuracy))
    print("Число не распознанных изображений = " + str(len(MISCLASSIFIED)) + ' из ' + str(total))


if __name__ == '__main__':
    main()
