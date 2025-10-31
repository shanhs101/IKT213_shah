import cv2
import numpy as np
from matplotlib import pyplot as plt


def HCD(reference_img):
    gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    reference_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite("harris.png", reference_img)
    cv2.imshow('dst', reference_img)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def SIFT(reference_image, image_to_align, max_features, good_match_precent):

    if len(reference_image.shape) == 3:
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_image

    if len(image_to_align.shape) == 3:
        image_to_align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
        image_to_align_color = image_to_align
    else:
        image_to_align_gray = image_to_align
        image_to_align_color = cv2.cvtColor(image_to_align, cv2.COLOR_GRAY2BGR)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference_gray, None)
    kp2, des2 = sift.detectAndCompute(image_to_align_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) >= max_features:
        print(f"Found {len(good)} good matches (threshold: {max_features})")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = reference_gray.shape
        aligned_img = cv2.warpPerspective(image_to_align_color, M, (w, h))

        cv2.imwrite('aligned.jpg', aligned_img)
        print("Aligned image saved as 'aligned.jpg'")

    else:
        print(f"Not enough matches found - {len(good)}/{max_features}")
        matchesMask = None
        aligned_img = None

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
    )

    matches_img = cv2.drawMatches(
        reference_gray, kp1,
        image_to_align_gray, kp2,
        good, None,
        **draw_params
    )

    cv2.imwrite('matches.jpg', matches_img)
    print("Matches visualization saved as 'matches.jpg'")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.imshow(matches_img)
    plt.title('Feature Matches')
    plt.axis('off')

    if aligned_img is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        plt.title('Aligned Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return aligned_img, matches_img


def main():
    reference_image = cv2.imread("reference_img.png")
    image_to_align = cv2.imread("align_this.jpg")

    max_features = 10
    good_match_percent = 0.7

    # HCD(reference_image)

    SIFT(reference_image, image_to_align, max_features, good_match_percent)


if __name__ == "__main__":
    main()