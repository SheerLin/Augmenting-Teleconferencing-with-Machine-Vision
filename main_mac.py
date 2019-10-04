import cv2


def show_webcam(mirror=False, src=0):
    cam = cv2.VideoCapture(src)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        if ret_val:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)

            # if cv2.waitKey(1) == 27:
            #     break  # esc to quit

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True, src=0)


if __name__ == '__main__':
    main()
