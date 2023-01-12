import cv2
import numpy as np
import time
import os

iceAktar = True
if iceAktar:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)


# Bu 2 resmi yükleyin ve aynı boyutta yeniden boyutlandırın.
kalem_img = cv2.resize(cv2.imread('kalem.png', 1), (50, 50))
silgi_img = cv2.resize(cv2.imread('silgi.jpg', 1), (50, 50))

kernel = np.ones((5, 5), np.uint8) #birlerle dolu 5x5'lik bir çekirdek

# Pencere boyutunun ayarlanabilmesi
cv2.namedWindow('Sanal Yazi', cv2.WINDOW_NORMAL)
cv2.namedWindow('Tahta', cv2.WINDOW_NORMAL)


# Üzerine çizeceğimiz tuval bu
tahta = None
tahta = np.zeros((720, 1280, 3), np.uint8)


# Bir arka plan çıkarıcı
arkaPlanNesnesi = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Bu eşik arka planda bozulma miktarını belirler.
arklaPlanEsigi = 800

# Kalem mi yoksa silgi mi kullandığınızı söyleyen bir değişken.
gecis = 'Kalem'

# Bu değişken ile bir önceki geçiş arasındaki süreyi izleyeceğiz.
sonGecis = time.time()

# x1,y1 noktalarını başlat
x1, y1 = 0, 0

# Gürültü eşiği
gurultu = 1000

#  Silecek için eşik, tuvali temizlememiz için konturun boyutu bundan daha büyük olmalıdır
silici = 40000

# Tuvali ne zaman temizleyeceğini söyleyen bir değişken
temizle = False

while (1):
    _, pencere = cap.read()
    pencere = cv2.flip(pencere, 1)

    # Tuvali siyah bir görüntü olarak başlat
    if tahta is None:
        tahta = np.zeros_like(pencere)

    # Çerçevenin sol üst köşesini alın ve orada arka plan çıkarıcı uygulanır
    solUst = pencere[0: 50, 0: 50]
    maske1 = arkaPlanNesnesi.apply(solUst)

    # Beyaz olan piksel sayısına dikkat edilir, bu bozulma düzeyidir.
    gecis = np.sum(maske1 == 255)

    # Kesinti, arka plan eşiğinden büyükse ve önceki geçişten sonra bir süre geçtiyse
    # bu normaldir. nesne türünü değiştirebilir.
    if gecis > arklaPlanEsigi and (time.time() - sonGecis) > 1:

        # Anahtarın zamanını kaydedin.
        sonGecis = time.time()

        if gecis == 'Kalem':
            gecis = 'Silgi'
        else:
            gecis = 'Kalem'

    # BGR'yi HSV'ye Dönüştür
    hsv = cv2.cvtColor(pencere, cv2.COLOR_BGR2HSV)

    # Bellekten okuyorsanız, üst ve alt aralıkları oradan yüklenir
    if iceAktar:
        alt_aralik = penval[0]
        usr_aralik = penval[1]

    # Aksi takdirde, üst ve alt aralık için kendi özel değerlerinizi tanımlayın.
    else:
        alt_aralik = np.array([26, 80, 147])
        usr_aralik = np.array([81, 255, 255])

    maske = cv2.inRange(hsv, alt_aralik, usr_aralik)

    # Gürültüden kurtulmak için morfolojik işlemler gerçekleştirin
    maske = cv2.erode(maske, kernel, iterations=1)
    maske = cv2.dilate(maske, kernel, iterations=2)

    # Konturları Bul
    kontur, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bir kontur olduğundan ve boyutunun olduğundan daha büyük olduğunda.

    if kontur and cv2.contourArea(max(kontur, key=cv2.contourArea)) > gurultu:

        c = max(kontur, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        # Kontur alanını alın
        alan = cv2.contourArea(c)

        # Daha önce herhangi bir nokta yoksa, tespit edilen x2,y2'yi kaydedin
        # koordinatları x1,y1 olarak.
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2

        else:
            if gecis == 'Kalem':
                # Tuval üzerine çizgiyi çizin
                tahta = cv2.line(tahta, (x1, y1),(x2, y2), [255, 0, 0], 5)
                tahta = cv2.line(tahta, (x1, y1), (x2, y2), [255, 0, 0], 5)

            else:
                cv2.circle(tahta, (x2, y2), 20,(0, 0, 0), -1)
                cv2.circle(tahta, (x2, y2), 20,(0, 0, 0), -1)

        # Çizgi çizildikten sonra yeni noktalar önceki noktalar olur.
        x1, y1 = x2, y2

        # Alan silecek eşiğinden büyükse,değişkeni True olarak değiştir
        if alan > silici:
            cv2.putText(tahta, 'Temizleniyor', (0, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
            temizle = True

    else:
        # Kontur tespit edilmediyse x1,y1 = 0
        x1, y1 = 0, 0

    # Bu kod parçası sadece düzgün çizim içindir. (İsteğe bağlı)
    _, maske = cv2.threshold(cv2.cvtColor(tahta, cv2.COLOR_BGR2GRAY), 20,
                            255, cv2.THRESH_BINARY)
    onPlan = cv2.bitwise_and(tahta, tahta, maske=maske)
    arkaPlan = cv2.bitwise_and(pencere, pencere,
                                 maske=cv2.bitwise_not(maske))
    pencere = cv2.add(onPlan, arkaPlan)

    # Kullandığımız şeye, kaleme veya silgiye bağlı olarak görüntüleri değiştirin.
    if gecis != 'Kalem':
        cv2.circle(pencere, (x1, y1), 20, (255, 255, 255), -1)
        pencere[0: 50, 0: 50] = silgi_img
    else:
        pencere[0: 50, 0: 50] = kalem_img

    cv2.imshow('Sanal Yazi', pencere)
    cv2.imshow('Tahta', tahta)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # temizle değişkeni true ise tuvali 1 saniye sonra temizle
    if temizle == True:
        time.sleep(1)
        tahta = None

        # Ve sonra temizle olarak false olarak ayarla
        temizle = False

cv2.destroyAllWindows()
cap.release()