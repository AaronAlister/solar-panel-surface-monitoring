import os
import csv
import argparse
from typing import cast
import cv2
import numpy as np


def put_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = 0.5,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = max(0, x - 3)
    y1 = max(0, y - th - base - 5)
    x2 = min(image.shape[1] - 1, x + tw + 6)
    y2 = min(image.shape[0] - 1, y + 4)
    cv2.rectangle(image, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(image, text, (x, y - 3), font, font_scale, fg, thickness, cv2.LINE_AA)


def classify_severity(defect_percent, low_thresh=2.0, med_thresh=7.0):
    if defect_percent < low_thresh:
        return "Low"
    if defect_percent < med_thresh:
        return "Medium"
    return "High"


def is_image_file(name):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return name.lower().endswith(exts)


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask cannot be None")
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return b


def extract_panel_roi(image: np.ndarray, roi_mask: np.ndarray | None = None):
    """
    Returns image ROI and panel pixel count.
    If roi_mask is None, full image is used.
    """
    if roi_mask is None:
        h, w = image.shape[:2]
        return image.copy(), None, h * w

    roi_mask = ensure_binary(roi_mask)
    panel_pixels = cv2.countNonZero(roi_mask)
    if panel_pixels == 0:
        h, w = image.shape[:2]
        return image.copy(), None, h * w

    roi_img = cv2.bitwise_and(image, image, mask=roi_mask)
    return roi_img, roi_mask, panel_pixels


def detect_defects(
    image: np.ndarray,
    roi_mask: np.ndarray | None = None,
    min_area: int = 20,
    canny_low: int = 50,
    canny_high: int = 150,
):
    roi_img, roi_mask_bin, panel_pixels = extract_panel_roi(image, roi_mask)

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    edges = cv2.Canny(enhanced, canny_low, canny_high)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    defect_mask = cv2.bitwise_or(edges_closed, thresh_clean)

    if roi_mask_bin is not None:
        defect_mask = cv2.bitwise_and(defect_mask, roi_mask_bin)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    filtered = np.zeros_like(defect_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    defect_mask = filtered

    defect_pixels = cv2.countNonZero(defect_mask)
    defect_percent = (defect_pixels / panel_pixels) * 100.0 if panel_pixels > 0 else 0.0

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        defect_count = len(areas)
        avg_area = float(np.mean(areas))
        max_area = float(np.max(areas))
    else:
        defect_count = 0
        avg_area = 0.0
        max_area = 0.0

    overlay = image.copy()
    overlay[defect_mask > 0] = (0, 0, 255)
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Draw bounding boxes and per-defect tags.
    for idx, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        area = int(cv2.contourArea(cnt))
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"Defect {idx} | area={area}px"
        put_label(result, label, x, max(20, y))

    summary = f"Defects={defect_count}  Severity={classify_severity(defect_percent)}  Defect%={defect_percent:.2f}"
    put_label(result, summary, 10, 24, font_scale=0.55)

    return {
        "defect_percent": defect_percent,
        "severity": classify_severity(defect_percent),
        "defect_mask": defect_mask,
        "overlay": result,
        "defect_count": defect_count,
        "avg_defect_area_px": avg_area,
        "max_defect_area_px": max_area,
        "panel_pixels": int(panel_pixels),
        "defect_pixels": int(defect_pixels),
    }


def make_viz_panel(original: np.ndarray, defect_mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Create a side-by-side visualization: original | mask | overlay."""
    orig = original.copy()
    mask_bgr = cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR)
    over = overlay.copy()

    put_label(orig, "Original Image", 10, 24, font_scale=0.6)
    put_label(mask_bgr, "Defect Mask: White=Detected Defect", 10, 24, font_scale=0.6)
    put_label(over, "Overlay: Red=Defect | Yellow Box=Region | Tag=ID+Area", 10, 24, font_scale=0.6)

    return cv2.hconcat([orig, mask_bgr, over])


def default_roi_mask_path(roi_dir, image_filename):
    stem, _ = os.path.splitext(image_filename)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = os.path.join(roi_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def process_folder(input_dir, output_dir, roi_dir=None, csv_name="results.csv", save_viz=False):
    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
        print("Create it or pass --input <folder> with your images.")
        return
    if roi_dir and not os.path.isdir(roi_dir):
        print(f"ROI folder not found: {roi_dir}")
        print("Proceeding without ROI masks.")
        roi_dir = None

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    if save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    rows = []
    files = sorted([f for f in os.listdir(input_dir) if is_image_file(f)])
    if not files:
        print(f"No image files found in: {input_dir}")
        print("Supported extensions: .jpg .jpeg .png .bmp .tif .tiff .webp")

    for fname in files:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            rows.append([fname, "", "", "", "", "", "READ_ERROR"])
            continue

        roi_mask = None
        if roi_dir:
            roi_path = default_roi_mask_path(roi_dir, fname)
            if roi_path:
                roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        out = detect_defects(img, roi_mask=roi_mask)

        stem, _ = os.path.splitext(fname)
        mask_path = os.path.join(mask_dir, f"{stem}_mask.png")
        overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")
        cv2.imwrite(mask_path, out["defect_mask"])
        cv2.imwrite(overlay_path, out["overlay"])
        if save_viz:
            viz = make_viz_panel(img, out["defect_mask"], out["overlay"])
            viz_path = os.path.join(viz_dir, f"{stem}_viz.png")
            cv2.imwrite(viz_path, viz)

        rows.append([
            fname,
            f"{out['defect_percent']:.2f}",
            out["severity"],
            str(out["defect_count"]),
            f"{out['avg_defect_area_px']:.2f}",
            f"{out['max_defect_area_px']:.2f}",
            "OK",
        ])

    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "defect_percent",
            "severity",
            "defect_count",
            "avg_defect_area_px",
            "max_defect_area_px",
            "status",
        ])
        writer.writerows(rows)

    print(f"Processed: {len(files)} images")
    print(f"CSV: {csv_path}")
    print(f"Masks: {mask_dir}")
    print(f"Overlays: {overlay_dir}")
    if save_viz:
        guide_path = os.path.join(output_dir, "visualization_guide.txt")
        with open(guide_path, "w", encoding="utf-8") as gf:
            gf.write("Visualization guide\n")
            gf.write("- Original Image: source image before detection.\n")
            gf.write("- Defect Mask (white/black): white pixels are detected defect regions.\n")
            gf.write("- Overlay: red highlight marks defect pixels, yellow rectangles are defect bounding boxes,\n")
            gf.write("  and each tag shows defect ID and area in pixels.\n")
        print(f"Visualizations: {viz_dir}")
        print(f"Guide: {guide_path}")


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = ensure_binary(pred_mask)
    gt = ensure_binary(gt_mask)
    inter = cv2.countNonZero(cv2.bitwise_and(pred, gt))
    union = cv2.countNonZero(cv2.bitwise_or(pred, gt))
    return (inter / union) if union > 0 else 1.0


def precision_recall_f1(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = ensure_binary(pred_mask)
    gt = ensure_binary(gt_mask)

    pred_bool = pred > 0
    gt_bool = gt > 0

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calibrate_thresholds(images_dir, labels_dir, roi_dir=None, min_area=20):
    if not os.path.isdir(images_dir):
        print(f"Input folder not found: {images_dir}")
        return None
    if not os.path.isdir(labels_dir):
        print(f"Labels folder not found: {labels_dir}")
        return None
    if roi_dir and not os.path.isdir(roi_dir):
        print(f"ROI folder not found: {roi_dir}")
        print("Proceeding without ROI masks for calibration.")
        roi_dir = None

    files = sorted([f for f in os.listdir(images_dir) if is_image_file(f)])

    low_grid = [30, 50, 70]
    high_grid = [120, 150, 180]
    best = None

    for low in low_grid:
        for high in high_grid:
            if high <= low:
                continue

            ious = []
            f1s = []
            used = 0

            for fname in files:
                img_path = os.path.join(images_dir, fname)
                label_path = default_roi_mask_path(labels_dir, fname)
                if label_path is None:
                    continue

                img = cv2.imread(img_path)
                gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if img is None or gt is None:
                    continue
                gt = cast(np.ndarray, gt)

                roi_mask = None
                if roi_dir:
                    rp = default_roi_mask_path(roi_dir, fname)
                    if rp:
                        roi_mask = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)

                out = detect_defects(
                    img,
                    roi_mask=roi_mask,
                    min_area=min_area,
                    canny_low=low,
                    canny_high=high,
                )
                pred = out["defect_mask"]

                # Apply ROI constraint to GT too for fair comparison.
                if roi_mask is not None:
                    roi_bin = ensure_binary(roi_mask)
                    gt = cv2.bitwise_and(gt, roi_bin)

                ious.append(iou_score(pred, gt))
                _, _, f1 = precision_recall_f1(pred, gt)
                f1s.append(f1)
                used += 1

            if used == 0:
                continue

            mean_iou = float(np.mean(ious))
            mean_f1 = float(np.mean(f1s))

            candidate = {
                "canny_low": low,
                "canny_high": high,
                "mean_iou": mean_iou,
                "mean_f1": mean_f1,
                "samples": used,
            }

            if best is None or (candidate["mean_f1"], candidate["mean_iou"]) > (
                best["mean_f1"],
                best["mean_iou"],
            ):
                best = candidate

    return best


def build_parser():
    p = argparse.ArgumentParser(description="Solar panel defect batch processor")
    p.add_argument("--input", default="input_images", help="Input image folder (default: input_images)")
    p.add_argument("--output", default="output_results", help="Output folder (default: output_results)")
    p.add_argument("--roi", default=None, help="Optional ROI mask folder")
    p.add_argument("--viz", action="store_true", help="Save side-by-side visualization images")
    sub = p.add_subparsers(dest="mode", required=False)

    run = sub.add_parser("run", help="Batch process images and export CSV/masks/overlays")
    run.add_argument("--input", default="input_images", help="Input image folder (default: input_images)")
    run.add_argument("--output", default="output_results", help="Output folder (default: output_results)")
    run.add_argument("--roi", default=None, help="Optional ROI mask folder")
    run.add_argument("--viz", action="store_true", help="Save side-by-side visualization images")

    cal = sub.add_parser("calibrate", help="Calibrate Canny thresholds using labeled masks")
    cal.add_argument("--input", required=True, help="Input image folder")
    cal.add_argument("--labels", required=True, help="Ground-truth defect mask folder")
    cal.add_argument("--roi", default=None, help="Optional ROI mask folder")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode in (None, "run"):
        process_folder(args.input, args.output, roi_dir=args.roi, save_viz=args.viz)
        return

    if args.mode == "calibrate":
        best = calibrate_thresholds(args.input, args.labels, roi_dir=args.roi)
        if not best:
            print("No valid labeled samples found for calibration.")
            return

        print("Best calibration:")
        print(f"  canny_low:  {best['canny_low']}")
        print(f"  canny_high: {best['canny_high']}")
        print(f"  mean_iou:   {best['mean_iou']:.4f}")
        print(f"  mean_f1:    {best['mean_f1']:.4f}")
        print(f"  samples:    {best['samples']}")


if __name__ == "__main__":
    main()


