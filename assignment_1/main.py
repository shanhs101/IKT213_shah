import cv2

def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Image Size:", image.size)
    print("Image Datatype:", image.dtype)


def record_camera():
    cam = cv2.VideoCapture(0)

    # Try to request 60 fps
    cam.set(cv2.CAP_PROP_FPS, 60)

    # Read back what the camera actually provides
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Requested 60 FPS, actual FPS:", fps)


    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS)

    with open("solutions/camera_outputs.txt", "w") as f:
        f.write(f"Width: {frame_width}\n")
        f.write(f"Height: {frame_height}\n")
        f.write(f"FPS: {fps}\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        out.write(frame)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    record_camera()

    image = cv2.imread("lena-1.png")
    if image is None:
        print("Error: Image file not found!")
    else:
        print_image_information(image)


if __name__ == "__main__":
    main()
