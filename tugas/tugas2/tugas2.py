import cv2
import numpy as np

# import gambar
img = cv2.imread("tugas/tugas2/tugas2.jpg")

# blur
img_blur = cv2.GaussianBlur(img, (3, 3), 100)

# canny edge detection algorithm
canny = cv2.Canny(img_blur, 125, 175)

# contours
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

selected_contour = contours[7]

cv2.drawContours(img, contours, 7, (0, 0, 0), 3)

# Tambahkan teks jumlah sisi di atas kiri gambar
epsilon = 0.02 * cv2.arcLength(selected_contour, True)
approx = cv2.approxPolyDP(selected_contour, epsilon, True)

cv2.putText(img, str(len(approx)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX , 5, (0, 0, 0), 2, cv2.LINE_AA) 

# Tampilkan gambar dengan kontur dan teks jumlah sisi
cv2.imshow("Detected Object with Contour", img)
cv2.imshow("canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
