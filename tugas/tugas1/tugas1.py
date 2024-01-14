import cv2
import numpy as np

# Baca gambar
image = cv2.imread('tugas/tugas1/tugas1.png')

# Konversi gambar ke format HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Tentukan rentang warna hijau dalam format HSV
lower_green = np.array([70, 50, 0])
upper_green = np.array([86, 255, 255])

# Buat mask untuk mengambil warna hijau
mask = cv2.inRange(hsv, lower_green, upper_green)

# Temukan kontur pada gambar
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterasi melalui semua kontur dan buat bounding box berwarna merah
for contour in contours:
    # Dapatkan bounding box untuk setiap kontur
    x, y, w, h = cv2.boundingRect(contour)
    
    # Gambar bounding box berwarna merah pada gambar asli
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Tampilkan gambar asli dengan bounding box berwarna merah
cv2.imshow('Deteksi Warna Hijau', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
