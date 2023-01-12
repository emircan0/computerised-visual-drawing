import cv2
import mediapipe as mp
import math

def main():
    kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    dedektor = ElYakala(min_algilama_guveni=0.5, maxEl=2)

    while True:
        success, img = kamera.read()
        img = cv2.flip(img, 1)
        # El ve yer işaretlerini bulun
        eller, img = dedektor.elBul(img)

        if eller:
            # el 1
            el_1 = eller[0]
            lmListesi = el_1["lmListesi"]  # 21 dönüm noktası listesi

            if len(eller) == 2:
                # el 2
                el_2 = eller[1]
                lmListesi2 = el_2["lmListesi"]  # 21 dönüm noktası listesi

                # # İki Yer İşareti Arasındaki Mesafeyi Bulun. Aynı el veya farklı eller olabilir
                boyut, bilgi, img = dedektor.konum(lmListesi[8][0:2], lmListesi2[8][0:2], img)  # with draw

        # Ekran
        cv2.imshow("Pencere", img)
        cv2.waitKey(1)


class ElYakala():
    def __init__(self, mode=False, maxEl=2, min_algilama_guveni=0.5, min_izleme_guveni=0.5):
        self.mode = mode
        self.maxEl = maxEl
        self.min_algilama_guveni = min_algilama_guveni
        self.min_izleme_guveni = min_izleme_guveni
        self.mpEller = mp.solutions.hands
        self.mpEller = mp.solutions.hands
        self.eller = self.mpEller.Hands(static_image_mode=self.mode, max_num_hands=self.maxEl,
                                        min_detection_confidence=self.min_algilama_guveni,
                                        min_tracking_confidence=self.min_izleme_guveni)
        self.mpCizim = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.parmaklar = []
        self.lmListesi = []

    def elBul(self, img, cizim=True, secim=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.eller.process(imgRGB)
        tumEller = []
        yukseklik, en, kanal = img.shape
        if self.results.multi_hand_landmarks:
            for elTipi, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                ellerim = {}
                ## lmListesi
                lmListesi = []
                xListesi = []
                yListesi = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * en), int(lm.y * yukseklik), int(lm.z * en)
                    lmListesi.append([px, py, pz])
                    xListesi.append(px)
                    yListesi.append(py)

                #KUTU
                xmin, xmax = min(xListesi), max(xListesi)
                ymin, ymax = min(yListesi), max(yListesi)
                kutuEn, kutuYukseklik = xmax - xmin, ymax - ymin
                kutu = xmin, ymin, kutuEn, kutuYukseklik
                cx, cy = kutu[0] + (kutu[2] // 2), \
                         kutu[1] + (kutu[3] // 2)

                ellerim["lmListesi"] = lmListesi
                ellerim["kutu"] = kutu
                ellerim["center"] = (cx, cy)

                if secim:
                    if elTipi.classification[0].label == "Right":
                        ellerim["type"] = "SAG EL"
                    else:
                        ellerim["type"] = "SOL EL"
                else:
                    ellerim["type"] = elTipi.classification[0].label
                tumEller.append(ellerim)

                # cizim
                if cizim:
                    self.mpCizim.draw_landmarks(img, handLms, self.mpEller.HAND_CONNECTIONS)
                    cv2.rectangle(img, (kutu[0] - 20, kutu[1] - 20),
                                  (kutu[0] + kutu[2] + 20, kutu[1] + kutu[3] + 20), (255, 0, 255), 2)
                    cv2.putText(img, ellerim["type"], (kutu[0] - 30, kutu[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        if cizim:
            return tumEller, img
        else:
            return tumEller

    def parmaklarHavada(self, ellerim):
        ellerimType = ellerim["type"]
        lmListesi = ellerim["lmListesi"]
        if self.results.multi_hand_landmarks:
            parmaklar = []
            # Baş parmak
            if ellerimType == "SAG EL":
                if lmListesi[self.tipIds[0]][0] < lmListesi[self.tipIds[0] - 1][0]:
                    parmaklar.append(1)
                else:
                    parmaklar.append(0)
            else:
                if lmListesi[self.tipIds[0]][0] > lmListesi[self.tipIds[0] - 1][0]:
                    parmaklar.append(1)
                else:
                    parmaklar.append(0)

            # 4 parmak
            for id in range(1, 5):
                if lmListesi[self.tipIds[id]][1] < lmListesi[self.tipIds[id] - 2][1]:
                    parmaklar.append(1)
                else:
                    parmaklar.append(0)
        return parmaklar

    def konum(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        boyut = math.hypot(x2 - x1, y2 - y1)
        bilgi = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return boyut, bilgi, img
        else:
            return boyut, bilgi


if __name__ == "__main__":
    main()
