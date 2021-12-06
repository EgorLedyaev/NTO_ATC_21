import numpy as np
import cv2


def one_hot_encode(label):

    """ Р¤СѓРЅРєС†РёСЏ РѕСЃСѓС‰РµСЃС‚РІР»СЏРµС‚ РїРµСЂРµРєРѕРґРёСЂРѕРІРєСѓ С‚РµРєСЃС‚РѕРІРѕРіРѕ "РЅР°Р·РІР°РЅРёСЏ" СЃРёРіРЅР°Р»Р°
     РІ СЃРїРёСЃРѕРє СЌР»РµРјРµРЅС‚РѕРІ, СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‰РёР№ РІС‹С…РѕРґРЅРѕРјСѓ СЃРёРіРЅР°Р»Сѓ

     Р’С…РѕРґРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹: С‚РµРєСЃС‚РѕРІР°СЏ РјРµС‚РєР°
     Р’С‹С…РѕРґРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹: РјРµС‚РєР° РІРІРёРґРµ СЃРїРёСЃРєР°

     РџСЂРёРјРµСЂ:
        one_hot_encode("red") РґРѕР»Р¶РЅРѕ РІРѕР·РІСЂР°С‰Р°С‚СЊ:        [1, 0, 0, 0, 0]
        one_hot_encode("yellow") РґРѕР»Р¶РЅРѕ РІРѕР·РІСЂР°С‰Р°С‚СЊ:     [0, 1, 0, 0, 0]
        one_hot_encode("green") РґРѕР»Р¶РЅРѕ РІРѕР·РІСЂР°С‰Р°С‚СЊ:      [0, 0, 1, 0, 0]
        one_hot_encode("yellow_red") РґРѕР»Р¶РЅРѕ РІРѕР·РІСЂР°С‰Р°С‚СЊ: [0, 0, 0, 1, 0]
        one_hot_encode("off") РґРѕР»Р¶РЅРѕ РІРѕР·РІСЂР°С‰Р°С‚СЊ:        [0, 0, 0, 0, 1]

     """
    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0, 0, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1, 0, 0]
    elif label == "yellow_red":
        one_hot_encoded = [0, 0, 0, 1, 0]
    elif label == "off":
        one_hot_encoded = [0, 0, 0, 0, 1]

    return one_hot_encoded


def standardize_input(image):
    """РџСЂРёРІРµРґРµРЅРёРµ РёР·РѕР±СЂР°Р¶РµРЅРёР№ Рє СЃС‚Р°РЅРґР°СЂС‚РЅРѕРјСѓ РІРёРґСѓ.
    Р’С…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ: РёР·РѕР±СЂР°Р¶РµРЅРёРµ (bgr)
    Р’С‹С…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ: СЃС‚Р°РЅРґР°СЂС‚РёР·РёСЂРѕРІР°РЅРЅРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёР№.
    """
    standard_im = image  # РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ, С„СѓРЅРєС†РёСЏ РЅРµ РјРµРЅСЏРµС‚ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ

    ## TODO: Р•СЃР»Рё РІС‹ С…РѕС‚РёС‚Рµ РїСЂРµРѕР±СЂР°Р·РѕРІР°С‚СЊ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РІ С„РѕСЂРјР°С‚, РѕРґРёРЅР°РєРѕРІС‹Р№ РґР»СЏ РІСЃРµС… РёР·РѕР±СЂР°Р¶РµРЅРёР№, СЃРґРµР»Р°Р№С‚Рµ СЌС‚Рѕ Р·РґРµСЃСЊ.
    return standard_im


def predict_label(rgb_image):
    """
     С„СѓРЅРєС†РёСЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ СЃРёРіРЅР°Р»Р° СЃРІРµС‚РѕС„РѕСЂР° РїРѕ РІС…РѕРґРЅРѕРјСѓ РёР·РѕР±СЂР°Р¶РµРЅРёСЋ

     Р’С…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ: РёР·РѕР±СЂР°Р¶РµРЅРёРµ (bgr)
     Р’С‹С…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ: РјРµС‚РєР° РІ С„РѕСЂРјР°С‚Рµ СЃРїРёСЃРєР° (СЃРјРѕС‚СЂРё one_hot_encode)

    """

    frame = cv2.resize(rgb_image, (60, 115))
    cutedFrame = frame[20:110, 8:55]

    hsv = cv2.cvtColor(cutedFrame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    red_sum = np.sum(v[0:27, 0:44])
    yellow_sum = np.sum(v[28:54, 0:44])
    green_sum = np.sum(v[55:81, 0:44])

    if ((min(red_sum, yellow_sum) / max(red_sum, yellow_sum)) * 100 >= 99 and (
            min(green_sum, yellow_sum) / max(green_sum, yellow_sum)) * 100 >= 99 and (
            (min(red_sum, green_sum) / max(red_sum, green_sum)) * 100 >= 99)):
        predicted_label = 'off'
    elif ((min(red_sum, yellow_sum) / max(red_sum,
                                          yellow_sum)) * 100 >= 89 and red_sum > green_sum and yellow_sum > green_sum):
        predicted_label = 'yellow_red'
    elif green_sum > yellow_sum and green_sum > red_sum:
        predicted_label = 'green'
    elif yellow_sum > green_sum and yellow_sum > red_sum:
        predicted_label = 'yellow'
    elif red_sum > green_sum and red_sum > yellow_sum:
        predicted_label = 'red'
    else:
        predicted_label = 'off'
    encoded_label = one_hot_encode(predicted_label)
    #print('red=', red_sum, 'yellow=', yellow_sum, 'green=', green_sum,'predict=',predicted_label)
    return encoded_label
