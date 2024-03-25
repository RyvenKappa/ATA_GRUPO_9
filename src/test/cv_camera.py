import cv2


capture = cv2.VideoCapture(0)#Capturar del dispositivo por defecto, la camara en este caso
ok = False
while(True):
    ok, img = capture.read()
    cv2.imshow("Camera",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()