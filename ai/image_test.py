import cv2

image = cv2.imread("test.jpg")

if image is None:
    print("Image could not be loaded.")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)

    cv2.imshow("Smart Event AI - Test Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()