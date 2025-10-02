import os
import cv2
import numpy as np

# --------- Paths----------
IMG1 = r"C:\Users\shanh\IKT213_shah\Figureprint\data_uia\UiA front1.png"
IMG2 = r"C:\Users\shanh\IKT213_shah\Figureprint\data_uia\UiA front3.jpg"
OUT_DIR = r"C:\Users\shanh\IKT213_shah\Figureprint\results_dataUIA"
os.makedirs(OUT_DIR, exist_ok=True)

# --------- Error checks / Helper ----------
def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def draw_matches(img1, img2, kp1, kp2, matches, mask=None, max_draw=200):

    if mask is not None and len(mask) == len(matches):
        inlier_matches = [m for m, keep in zip(matches, mask.ravel().tolist()) if keep]
        to_draw = inlier_matches[:max_draw] if inlier_matches else matches[:max_draw]
    else:
        to_draw = matches[:max_draw]

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis

def hstack_same_height(img_a, img_b):

    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        img_b = cv2.resize(img_b, (int(w2*scale), h1), interpolation=cv2.INTER_AREA)
    return np.hstack([img_a, img_b])

# --------- SIFT + FLANN ----------
def run_sift(img1, img2, ratio=0.75, checks=64):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return {
            "name": "SIFT+FLANN",
            "kp1": len(kp1), "kp2": len(kp2),
            "good": 0, "inliers": 0,
            "vis": draw_matches(img1, img2, kp1, kp2, []),
        }

    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:  # guard
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    H, mask, inliers = None, None, 0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
        if mask is not None:
            inliers = int(mask.sum())

    vis = draw_matches(img1, img2, kp1, kp2, good, mask)
    return {
        "name": "SIFT+FLANN",
        "kp1": len(kp1), "kp2": len(kp2),
        "good": len(good), "inliers": inliers,
        "vis": vis,
    }

# --------- ORB + BF ----------
def run_orb(img1, img2, ratio=0.7):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return {
            "name": "ORB+BF",
            "kp1": len(kp1), "kp2": len(kp2),
            "good": 0, "inliers": 0,
            "vis": draw_matches(img1, img2, kp1, kp2, []),
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    H, mask, inliers = None, None, 0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
        if mask is not None:
            inliers = int(mask.sum())

    vis = draw_matches(img1, img2, kp1, kp2, good, mask)
    return {
        "name": "ORB+BF",
        "kp1": len(kp1), "kp2": len(kp2),
        "good": len(good), "inliers": inliers,
        "vis": vis,
    }

# --------- Main Code ----------
if __name__ == "__main__":
    img1 = load_gray(IMG1)
    img2 = load_gray(IMG2)

    res_sift = run_sift(img1, img2, ratio=0.75, checks=64)
    res_orb  = run_orb(img1, img2, ratio=0.70)

    def report(res, inlier_threshold=15):
        decision = "MATCHED" if res["inliers"] >= inlier_threshold else "NOT MATCHED"
        print(f"\n[{res['name']}]")
        print(f"Keypoints: img1={res['kp1']}  img2={res['kp2']}")
        print(f"Good matches (ratio test): {res['good']}")
        print(f"Inlier matches (RANSAC):  {res['inliers']}")
        print(f"Decision (inliers >= {inlier_threshold}): {decision}")
        return decision

    dec_sift = report(res_sift, inlier_threshold=15)
    dec_orb  = report(res_orb,  inlier_threshold=15)

    sift_path = os.path.join(OUT_DIR, "UiA_SIFT_FLANN_matches.png")
    orb_path  = os.path.join(OUT_DIR, "UiA_ORB_BF_matches.png")
    cv2.imwrite(sift_path, res_sift["vis"])
    cv2.imwrite(orb_path,  res_orb["vis"])
    print(f"\nSaved:\n  {sift_path}\n  {orb_path}")

    combo = hstack_same_height(res_sift["vis"], res_orb["vis"])

    top = np.zeros((60, combo.shape[1], 3), dtype=np.uint8)
    combo_annot = np.vstack([top, combo])
    cv2.putText(combo_annot, "SIFT + FLANN  (" + dec_sift + ")", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(combo_annot, "ORB + BF  (" + dec_orb + ")", (combo.shape[1]//2 + 40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    combo_path = os.path.join(OUT_DIR, "UiA_SideBySide_SIFT_vs_ORB.png")
    cv2.imwrite(combo_path, combo_annot)
    print(f"Saved side-by-side: {combo_path}")

    cv2.imshow("SIFT vs ORB", combo_annot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
