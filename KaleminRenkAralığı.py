import cv2
import numpy as np
import time

# İzleme çubuğu işlevine giren gerekli bir geri arama yöntemi.
def nothing(x):
    pass

# Web kamerası başlatılıyor.
kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)
kamera.set(4, 720)

# İzleme çubukları adında bir pencere oluşturun.
cv2.namedWindow("Pencere")

# Şimdi H,S ve V kanallarının alt ve üst aralığını kontrol edecek 6 izleme çubuğu oluşturun.
# Argümanlar şu şekildedir: İzleme çubuğunun adı, pencere adı, aralık, geri arama işlevi.
# Hue için aralık 0-179 ve S,V için 0-255'tir.
cv2.createTrackbar("L - H", "Pencere", 0, 179, nothing)
cv2.createTrackbar("L - S", "Pencere", 0, 255, nothing)
cv2.createTrackbar("L - V", "Pencere", 0, 255, nothing)
cv2.createTrackbar("U - H", "Pencere", 179, 179, nothing)
cv2.createTrackbar("U - S", "Pencere", 255, 255, nothing)
cv2.createTrackbar("U - V", "Pencere", 255, 255, nothing)

while True:

    # Web kamerası kare kare okumaya başlar.
    ret, pencere = kamera.read()
    if not ret:
        break
    # Çerçeveyi yatay olarak çevriliyor (Gerekli değil)
    pencere = cv2.flip(pencere, 1)

    # BGR görüntüsünü HSV görüntüsüne dönüştürülüyor.
    hsv = cv2.cvtColor(pencere, cv2.COLOR_BGR2HSV)

    # Kullanıcı bunları değiştirdikçe izleme çubuğunun yeni değerlerini gerçek zamanlı olarak alınır
    l_h = cv2.getTrackbarPos("L - H", "Pencere")
    l_s = cv2.getTrackbarPos("L - S", "Pencere")
    l_v = cv2.getTrackbarPos("L - V", "Pencere")
    u_h = cv2.getTrackbarPos("U - H", "Pencere")
    u_s = cv2.getTrackbarPos("U - S", "Pencere")
    u_v = cv2.getTrackbarPos("U - V", "Pencere")

    # İzleme çubuğu tarafından seçilen değere göre alt ve üst HSV aralığını ayarlayın
    alt_aralik = np.array([l_h, l_s, l_v])
    ust_aralik = np.array([u_h, u_s, u_v])

    # Görüntüyü filtreleyin
    maske = cv2.inRange(hsv, alt_aralik, ust_aralik)

    # Hedef rengin gerçek kısmını da görselleştirebilirsiniz (Opsiyonel)
    res = cv2.bitwise_and(pencere, pencere, mask=maske)

    # İkili maskeyi 3 kanal görüntüsüne dönüştürmek
    mask_3 = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)

    # maskeyi, orijinal çerçeveyi ve filtrelenmiş sonucu kaydedilir
    stack = np.hstack((mask_3, pencere, res))

    # Bu yığılmış çerçeveyi boyutun %40'ında göster.
    cv2.imshow('Pencere', cv2.resize(stack, None, fx=0.4, fy=0.4))

    # Kullanıcı ESC'ye basarsa programdan çıkın
    key = cv2.waitKey(1)
    if key == 27:
        break

    # Kullanıcı `s` tuşuna basarsa bu diziyi yazdırın.
    if key == ord('s'):
        dizi = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(dizi)

        # Ayrıca bu diziyi penval.npy olarak kaydedin
        np.save('penval', dizi)
        break

# Kamerayı kapat ve pencereleri yok et.
kamera.release()
cv2.destroyAllWindows()