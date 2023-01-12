import cv2
import numpy as np
import time

# Bu değişken, renk aralığını bellekten yüklemek isteyip istemediğimizi belirler.
# veya not defterinde tanımlananları kullanın.
iceAktar = True

# Doğruysa, bellekten renk aralığını yükleyin
if iceAktar:
    penval = np.load('penval.npy')

kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)
kamera.set(4, 720)

# morfolojik işlemler için çekirdek
kernel = np.ones((5, 5), np.uint8)

# Bu eşik gürültüyü filtrelemek için kullanılır, gerçek bir kontur olarak
# nitelendirilmesi için kontur alanı bundan daha büyük olmalıdır.
gurultu = 500

while (1):

    _, pencere = kamera.read()
    pencere = cv2.flip(pencere, 1)

    # BGR'yi HSV'ye Dönüştür
    hsv = cv2.cvtColor(pencere, cv2.COLOR_BGR2HSV)

    # Eğer hafızadan okuyorsanız, üst ve alt açıları oradan yükleyin.
    if iceAktar:
        alt_aralik = penval[0]
        ust_aralik = penval[1]

    # Aksi takdirde, üst ve alt aralık için kendi özel değerlerinizi tanımlayın.
    else:
        alt_aralik = np.array([26, 80, 147])
        ust_aralik = np.array([81, 255, 255])

    maske = cv2.inRange(hsv, alt_aralik, ust_aralik)

    # Gürültüden kurtulmak için morfolojik işlemleri gerçekleştirin
    maske = cv2.erode(maske, kernel, iterations=1)  # Nesne alanını artırır ve özellikleri vurgulamak için kullanılır
    maske = cv2.dilate(maske, kernel, iterations=2) # Küçük beyaz gürültüleri gidermek ve birbirine bağlı iki nesneyi ayırmak için kullanılır.

    # Çerçevedeki Konturları Bul.
    kontur, hiyerarsi = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bir kontur olduğundan ve boyutunun gürültü eşiğinden büyük olduğundan emin olun.
    if kontur and cv2.contourArea(max(kontur, key=cv2.contourArea)) > gurultu:
        # Alana göre en büyük konturu yakala
        c = max(kontur, key=cv2.contourArea)

        # Bu kontur etrafındaki sınırlayıcı kutu koordinatlarını alın
        x, y, w, h = cv2.boundingRect(c) #ikili görüntünün etrafına yaklaşık bir dikdörtgen çizmek için kullanılır.
        # Bu işlev, bir görüntüden konturlar elde edildikten sonra ilgilenilen bölgeyi vurgulamak için kullanılır.

        # O sınırlayıcı kutuyu çiz
        cv2.rectangle(pencere, (x, y), (x + w, y + h), (0, 25, 255), 2)

    cv2.imshow('pencere', pencere)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
kamera.release()