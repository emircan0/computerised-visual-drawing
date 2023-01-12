import ElModeli as elt
import cv2
import os
import numpy as np

# parametreler
genislik, yukseklik = 1280, 720
esikCizgisi = 300
sunumAdresi = "Sunum"

# Kamera Kurulumu
kamera = cv2.VideoCapture(0)
kamera.set(3, genislik)
kamera.set(4, yukseklik)

# El Dedektörü
elYakalayici = elt.ElYakala(min_algilama_guveni=0.5, maxEl=1)

# Değişkenler
gecikme = 30
gecis = False
sayac = 0
sunumNo = 0
notlar = [[]]
notNumarası = -1
notlarBaslat = False
hs, ws = int(120 * 1), int(213 * 1)  # küçük kamera genişliği ve yüksekliği

# Sunum görüntülerinin listesini alma işlemi
sunumlar = sorted(os.listdir(sunumAdresi), key=len)
print(sunumlar)

while True:
    # img çerçevesi al
    success, img = kamera.read()
    img = cv2.flip(img, 1)
    tumSunumlar = os.path.join(sunumAdresi, sunumlar[sunumNo])
    imgYeni = cv2.imread(tumSunumlar)

    # El ve yer işaretlerini bulun
    eller , img = elYakalayici.elBul(img)

    # Hareket Eşiği çizgisi çizin
    cv2.line(img, (0, esikCizgisi), (genislik, esikCizgisi), (0, 255, 0), 10)

    if eller and gecis is False: # El algılanırsa
        el = eller[0]
        cx, cy = el["center"]
        lmListesi = el["lmListesi"]  # 21 etiker listesi
        parmaklar = elYakalayici.parmaklarHavada(el)  # Hangi parmakların yukarıda olduğu listesi
        #print(parmaklar)

        # Daha kolay çizim için değerleri sınırlayın
        xVal = int(np.interp(lmListesi[8][0], [genislik // 2, genislik], [0, genislik]))
        yVal = int(np.interp(lmListesi[8][1], [150, yukseklik - 150], [0, yukseklik]))
        isaretParmagi = xVal, yVal

        if cy <= esikCizgisi:  # El yüzün hizasında ise
            if parmaklar == [0, 0, 0, 0, 1]:
                #print("SAG EL")
                gecis = True
                if sunumNo > 0:
                    sunumNo -= 1
                    notlar = [[]]
                    notNumarası = -1
                    notlarBaslat = False
            if parmaklar == [1, 0, 0, 0, 0]:
                #print("SOL EL")
                gecis = True
                if sunumNo < len(sunumlar) - 1:
                    sunumNo += 1
                    notlar = [[]]
                    notNumarası = -1
                    notlarBaslat = False

        if parmaklar == [0, 1, 1, 0, 0]:
            cv2.circle(imgYeni, isaretParmagi, 12, (0, 0, 255), cv2.FILLED)

        if parmaklar == [0, 1, 0, 0, 0]:
            if notlarBaslat is False:
                notlarBaslat = True
                notNumarası += 1
                notlar.append([])
            print(notNumarası)
            notlar[notNumarası].append(isaretParmagi)
            cv2.circle(imgYeni, isaretParmagi, 12, (0, 0, 255), cv2.FILLED)

        else:
            notlarBaslat = False

        if parmaklar == [0, 1, 1, 1, 0]:
            if notlar:
                notlar.pop(-1)
                notNumarası -= 1
                gecis = True

    else:
        notlarBaslat = False

    if gecis:
        sayac += 1
        if sayac > gecikme:
            sayac = 0
            gecis = False

    for i, dizi in enumerate(notlar):
        for j in range(len(dizi)):
            if j != 0:
                cv2.line(imgYeni, dizi[j - 1], dizi[j], (0, 0, 200), 12)

    kameraKose = cv2.resize(img, (ws, hs))
    h, w, _ = imgYeni.shape
    imgYeni[0:hs, w - ws: w] = kameraKose

    cv2.imshow("Sunum", imgYeni)
    #cv2.imshow("Pencere", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break