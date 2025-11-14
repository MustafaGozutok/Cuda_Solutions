#!/usr/bin/env python3
"""
dualfg_botsort_final.py

Temiz ve son sürüm:
 - masked_mog2 native extension (.so) kullanımı
 - Ultralytics YOLO + dahili BoT-SORT (ReID opsiyonlu) ile kişi takibi
 - Dual-FG (short via masked_mog2, long via basit background) + Abandoned object owner-mapping
 - Minimal bağımlılıklar: opencv, numpy, ultralytics
"""

from pathlib import Path
import argparse
import time
import math
import sys
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import cv2
from ultralytics import YOLO

# --------------------------
# masked_mog2 native extension (.so) yükleme
# --------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(THIS_DIR, "build")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import masked_mog2 as mm  # type: ignore

# --------------------------
# Utilities
# --------------------------
def iou_bbox(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    ua = w1 * h1 + w2 * h2 - inter
    return 0.0 if ua <= 0 else inter / ua


def clip_bbox(bbox, W, H):
    x, y, w, h = bbox
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - int(round(x))))
    h = max(1, min(h, H - int(round(y))))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]


def nms_boxes(boxes, scores=None, iou_thresh=0.45):
    """
    Basit NMS implementasyonu.
    boxes: [ [x,y,w,h], ... ]
    scores: opsiyonel, yoksa area kullanılır.
    """
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    areas = boxes_np[:, 2] * boxes_np[:, 3]

    if scores is None:
        scores = areas
    scores = np.array(scores, dtype=np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = rest[inds]

    return keep


def overlay_colored_mask(frame, mask, color, alpha=0.5):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[:] = color
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = frame.copy()
    idx = mask3[..., 0] > 0
    blended[idx] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[idx]
    return blended


def random_color_for_id(idx):
    np.random.seed(int(idx) + 12345)
    return tuple(int(v) for v in np.random.randint(0, 255, 3))


# --------------------------
# YOLO + BoT-SORT tracker wrapper
# --------------------------
class YoloBoTSORTTracker:
    """
    Ultralytics YOLO + dahili BoT-SORT (ReID opsiyonlu) için basit wrapper.

    - YOLO modelini yüklüyor.
    - tracker=botsort.yaml ile BoT-SORT'u aktive ediyor.
    - update(frame) -> [(track_id, [x,y,w,h], conf), ...] sadece person (cls==0)
    """

    def __init__(self, model_path, tracker_cfg=None, device="cpu", conf=0.35):
        self.model = YOLO(model_path)
        # Model device ayarı
        try:
            self.model.to(device)
        except Exception:
            pass
        self.tracker_cfg = tracker_cfg
        self.device = device
        self.conf = conf

        print(
            f"[INFO] YoloBoTSORTTracker initialized (model={model_path}, "
            f"tracker_cfg={tracker_cfg}, device={device}, conf={conf})"
        )

    def update(self, frame):
        """
        Tek kare için YOLO+BoTSORT çalıştırır.
        Dönen liste: [(track_id, [x,y,w,h], conf), ...]
        """
        # YOLO track: persist=True ile ID'ler kareler arası korunuyor.
        if self.tracker_cfg:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_cfg,
                conf=self.conf,
                verbose=False,
            )
        else:
            results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)

        if not results:
            return []

        res0 = results[0]
        boxes = res0.boxes

        if boxes is None or boxes.id is None:
            return []

        # CPU'ya çek
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()
        if boxes.cls is not None:
            cls = boxes.cls.int().cpu().numpy()
        else:
            cls = np.zeros_like(track_ids)
        if boxes.conf is not None:
            conf = boxes.conf.cpu().numpy()
        else:
            conf = np.ones_like(track_ids, dtype=float)

        tracks = []
        for bb, tid, c, cf in zip(xyxy, track_ids, cls, conf):
            # Sadece person (COCO class 0)
            if int(c) != 0:
                continue
            x1, y1, x2, y2 = bb
            x = int(max(0, math.floor(x1)))
            y = int(max(0, math.floor(y1)))
            w = int(max(1, math.ceil(x2 - x1)))
            h = int(max(1, math.ceil(y2 - y1)))
            tracks.append((int(tid), [x, y, w, h], float(cf)))

        return tracks


# --------------------------
# DualFG, AbandonedManager (aynı mantık, küçük cleanup)
# --------------------------
class DualFG:
    def __init__(self, model, W, H, C, short_lr=0.01, alpha_long=0.001, thresh_long=30, min_area=400, min_static_frames=90):
        self.model = model
        self.W = W
        self.H = H
        self.C = C
        self.short_lr = short_lr
        self.alpha_long = alpha_long
        self.thresh_long = thresh_long
        self.min_area = min_area
        self.min_static_frames = min_static_frames

        self.B_long = None
        self.static_counter = np.zeros((H, W), dtype=np.uint32)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def init_long_bg(self, frame):
        self.B_long = frame.astype(np.float32)

    def get_bg(self):
        try:
            if hasattr(self.model, "getBackgroundImage"):
                return self.model.getBackgroundImage()
            if hasattr(self.model, "get_bg"):
                return self.model.get_bg()
            return None
        except Exception:
            return None

    def apply_short(self, frame, roi_mask=None, protected_mask=None):
        temp = frame.copy()
        if protected_mask is not None and protected_mask.any():
            bg = self.get_bg()
            if bg is not None:
                mk = protected_mask > 0
                temp[mk] = bg[mk]

        try:
            mask = self.model.apply(temp, roi_mask if roi_mask is not None else None, None, float(self.short_lr), False)
        except TypeError:
            mask = self.model.apply(temp, roi_mask if roi_mask is not None else None, None, float(self.short_lr))

        if mask is None:
            mask = np.zeros((self.H, self.W), dtype=np.uint8)
        elif mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return mask

    def compute_long(self, frame, roi_mask=None):
        diff = cv2.absdiff(frame.astype(np.uint8), cv2.convertScaleAbs(self.B_long))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask_long = cv2.threshold(gray, self.thresh_long, 255, cv2.THRESH_BINARY)
        mask_long = cv2.morphologyEx(mask_long, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask_long = cv2.morphologyEx(mask_long, cv2.MORPH_CLOSE, self.kernel_big, iterations=1)
        if roi_mask is not None:
            mask_long = cv2.bitwise_and(mask_long, roi_mask)
        return mask_long

    def update_long(self, frame, protected_mask=None):
        if protected_mask is None:
            mask_update = np.ones((self.H, self.W), dtype=np.float32)[:, :, None]
        else:
            mask_update = (protected_mask == 0).astype(np.float32)[:, :, None]

        self.B_long = (
            self.B_long * (1.0 - (self.alpha_long * mask_update)) + frame.astype(np.float32) * (self.alpha_long * mask_update)
        )

    def find_static(self, mask_short, mask_long, roi_mask=None, person_mask=None):
            # long var, short yok → statik aday
        static_candidate = cv2.bitwise_and(mask_long, cv2.bitwise_not(mask_short))

        if roi_mask is not None:
            static_candidate = cv2.bitwise_and(static_candidate, roi_mask)

        # YENİ: insan piksellerini statik haritadan çıkar
        if person_mask is not None:
            static_candidate = cv2.bitwise_and(static_candidate, cv2.bitwise_not(person_mask))

        static_candidate = cv2.morphologyEx(static_candidate, cv2.MORPH_CLOSE, self.kernel_big, iterations=1)
        static_candidate = cv2.dilate(static_candidate, self.kernel, iterations=1)
        static_candidate = cv2.morphologyEx(static_candidate, cv2.MORPH_OPEN, self.kernel, iterations=1)

        self.static_counter = np.where(
            static_candidate > 0,
            np.minimum(self.static_counter + 1, 2**31 - 1),
            0,
        ).astype(np.uint32)

        ns, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(static_candidate, connectivity=8)
        dets_static = []
        for i in range(1, ns):
            x, y, w, h, area = stats_s[i]
            if area < self.min_area:
                continue
            dets_static.append((i, [int(x), int(y), int(w), int(h)]))
        return dets_static, labels_s, static_candidate


class AbandonedManager:
    def __init__(self, min_static_frames=90, conf_thresh=0.65, owner_left_sec=0.8, owner_lookback_sec=1.0, fps=25, person_overlap_thresh=0.25):
        self.blob_map = {}
        self.next_blob_id = 1
        self.min_static_frames = min_static_frames
        self.conf_thresh = conf_thresh
        self.owner_left_sec = owner_left_sec
        self.owner_lookback_sec = owner_lookback_sec
        self.fps = fps
        self.person_overlap_thresh = person_overlap_thresh

    def match_create(self, dets_static, labels_s, static_counter, tracks_dict, person_track_ids, frame_idx):
        for comp_idx, bbox in dets_static:
            region_mask = labels_s == comp_idx
            region_area = int(np.count_nonzero(region_mask))
            if region_area == 0:
                continue

            persist_med = int(np.median(static_counter[region_mask]))

            # Var olan blob ile overlap
            found_blob = None
            best = 0.0
            for bid, binfo in self.blob_map.items():
                pm = binfo.get("protected_mask")
                if pm is None:
                    continue
                inter = int(np.count_nonzero(np.logical_and(pm, region_mask)))
                if inter == 0:
                    continue
                overlap = inter / float(region_area)
                if overlap > best:
                    best = overlap
                    found_blob = bid

            if found_blob and best >= 0.45:
                self.blob_map[found_blob]["bbox"] = bbox
                self.blob_map[found_blob]["last_seen"] = frame_idx
                continue

            # IoU ile eşle
            if found_blob is None:
                for bid, binfo in self.blob_map.items():
                    pb = binfo.get("bbox")
                    if pb is None:
                        continue
                    if iou_bbox(pb, bbox) > 0.5:
                        found_blob = bid
                        break
                if found_blob:
                    self.blob_map[found_blob]["bbox"] = bbox
                    self.blob_map[found_blob]["last_seen"] = frame_idx
                    continue

            # Çok kısa süreli statikler elensin
            if persist_med < max(2, int(self.min_static_frames / 4)):
                continue

            # Owner mapping: kişi ile IoU
            owner_candidate = None
            for tid, bbox_t in tracks_dict.items():
                if tid not in person_track_ids:
                    continue
                if iou_bbox(bbox_t, bbox) > 0.05:
                    owner_candidate = tid
                    break

            bid = self.next_blob_id
            self.next_blob_id += 1
            prot_mask = np.zeros_like(static_counter, dtype=np.uint8)
            prot_mask[region_mask] = 1

            self.blob_map[bid] = {
                "bbox": bbox,
                "owner_id": owner_candidate,
                "status": "mapped" if owner_candidate else "unmapped",
                "protected_mask": prot_mask,
                "last_seen": frame_idx,
                "persist_med": persist_med,
            }

    def evaluate(self, frame, static_counter, person_mask, track_last_seen, frame_idx):
        out = []

        for bid, binfo in list(self.blob_map.items()):
            bbox = binfo["bbox"]
            x, y, w, h = bbox
            region_mask = binfo["protected_mask"].astype(bool)
            region_area = int(np.count_nonzero(region_mask))

            if region_area > 0:
                binfo["persist_med"] = int(np.median(static_counter[region_mask]))
            
            overlap_person = 0.0
            if person_mask is not None and person_mask.any() and region_area > 0:
                overlap_person = float(
                    np.count_nonzero(np.logical_and(region_mask, person_mask > 0))
                ) / float(region_area)

            # Eğer blob büyük ölçüde insanın üstündeyse → bu blob'u tamamen yok say
            if overlap_person > self.person_overlap_thresh:
                # insanlarla ilgili statik blob, candidate/abandoned istemiyoruz
                del self.blob_map[bid]
                continue

            owner = binfo.get("owner_id", None)
            status = binfo.get("status", "mapped")

            owner_left = True
            if owner is not None:
                last_seen = track_last_seen.get(owner, 0)
                if frame_idx - last_seen <= max(1, int(self.owner_left_sec * self.fps)):
                    owner_left = False

            if status == "abandoned":
                collected = False
                if person_mask.any():
                    overlap = float(np.count_nonzero(np.logical_and(region_mask, person_mask > 0))) / float(
                        max(1, region_area)
                    )
                    if overlap > 0.35:
                        collected = True
                if collected:
                    del self.blob_map[bid]
                    continue
                out.append((bid, bbox, "abandoned", 1.0, binfo["protected_mask"]))
                continue

            # Confidence hesaplama
            persist_norm = min(1.0, binfo.get("persist_med", 0) / float(max(1, self.min_static_frames)))
            area_norm = min(1.0, region_area / float(max(1, 100)))

            mask_uint8 = (region_mask.astype(np.uint8) * 255).astype(np.uint8)
            mean_inside = cv2.mean(frame, mask=mask_uint8)[:3]

            ring = cv2.dilate(mask_uint8, np.ones((7, 7), np.uint8))
            ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask_uint8))
            mean_ring = cv2.mean(frame, mask=ring)[:3] if ring.any() else mean_inside

            continuity = np.linalg.norm(np.array(mean_inside) - np.array(mean_ring))
            continuity_score = min(1.0, continuity / 60.0)

            owner_left_score = 1.0 if owner_left else 0.0

            w_p, w_a, w_o, w_c = 0.35, 0.15, 0.30, 0.20
            conf = w_p * persist_norm + w_a * area_norm + w_o * owner_left_score + w_c * continuity_score

            if conf >= self.conf_thresh and (binfo.get("persist_med", 0) >= self.min_static_frames):
                binfo["status"] = "abandoned"
                out.append((bid, bbox, "abandoned", conf, binfo["protected_mask"]))
            else:
                out.append((bid, bbox, "candidate", conf, binfo["protected_mask"]))

        return out


# --------------------------
# Argument parser & main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--mask", default=None)
    p.add_argument("--out", default="out_botsort_abandoned_final.mp4")
    p.add_argument("--display", action="store_true")

    # YOLO + tracker
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--tracker-config", default="botsort.yaml")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--device", default="cpu")
    p.add_argument("--with-reid", action="store_true", help="ReID: botsort.yaml içinde ayarla (with_reid: True)")
    p.add_argument("--reid-model", default=None, help="ReID modeli (botsort.yaml içindeki 'model' alanı)")

    # DualFG & abandoned params
    p.add_argument("--short-lr", type=float, default=0.01)
    p.add_argument("--alpha-long", type=float, default=0.001)
    p.add_argument("--thresh-long", type=int, default=30)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--min-static-frames", type=int, default=90)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--conf-thresh", type=float, default=0.65)
    p.add_argument("--owner-left-sec", type=float, default=0.8)

    return p.parse_args()


def main():
    args = parse_args()

    if args.with_reid:
        print(
            "[INFO] ReID is requested. Please ensure your tracker config (e.g. botsort.yaml) "
            "has 'with_reid: True' and 'model: auto' or a proper ReID model path."
        )
        if args.reid_model:
            print(f"[INFO] Suggested ReID model in YAML -> model: {args.reid_model}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: %s" % args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        C = int(cap.get(cv2.CAP_PROP_CHANNEL))
        if C <= 0:
            C = 3
    except Exception:
        C = 3

    # ROI mask
    roi_mask_uint8 = None
    if args.mask:
        roi = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise RuntimeError("Mask okunamadı: %s" % args.mask)
        if roi.shape[:2] != (H, W):
            roi = cv2.resize(roi, (W, H), interpolation=cv2.INTER_NEAREST)
        roi_mask_uint8 = (roi > 0).astype(np.uint8) * 255

    # masked_mog2 parametreleri
    params = mm.MOG2Params()
    params.history = 500
    params.nmixtures = 3
    params.varThreshold = 16.0
    params.varThresholdGen = 9.0
    params.varInit = 15.0
    params.varMin = 4.0
    params.varMax = 75.0
    params.backgroundRatio = 0.9
    params.detectShadows = False
    params.shadowValue = 127
    params.tau = 0.5
    params.motionThreshold = 20

    model = mm.MaskedMOG2(params)
    model.initialize(W, H, C, W * C)

    # DualFG init
    dual = DualFG(
        model,
        W,
        H,
        C,
        short_lr=args.short-lr if hasattr(args, "short-lr") else args.short_lr,
        alpha_long=args.alpha_long,
        thresh_long=args.thresh_long,
        min_area=args.min_area,
        min_static_frames=args.min_static_frames,
    )

    # İlk frame ile long BG init
    ret, f0 = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")
    dual.init_long_bg(f0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # YOLO + BoTSORT tracker
    yolo_tracker = YoloBoTSORTTracker(
        model_path=args.yolo_model,
        tracker_cfg=args.tracker_config,
        device=args.device,
        conf=args.yolo_conf,
    )

    abandoned = AbandonedManager(
        min_static_frames=args.min_static_frames,
        conf_thresh=args.conf_thresh,
        owner_left_sec=args.owner_left_sec,
        fps=fps,
    )

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    frame_idx = 0
    start_time = time.time()
    track_last_seen = {}

    print("[INFO] Starting main loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_proc = frame.copy()
        frame_masked = cv2.bitwise_and(frame, frame, mask=roi_mask_uint8) if roi_mask_uint8 is not None else frame

        # 1) Protected mask (abandoned blob'lar için)
        combined_prot = np.zeros((H, W), dtype=np.uint8)
        for b in abandoned.blob_map.values():
            if b["status"] in ("mapped", "abandoned") and b.get("protected_mask") is not None:
                combined_prot = np.logical_or(combined_prot, b["protected_mask"])
        combined_prot_uint8 = (combined_prot.astype(np.uint8) * 255).astype(np.uint8)

        # 2) Short / Long FG
        mask_short = dual.apply_short(frame_masked, roi_mask=roi_mask_uint8, protected_mask=combined_prot_uint8)
        mask_long  = dual.compute_long(frame_masked, roi_mask=roi_mask_uint8)

        # 3) YOLO + BoTSORT tracker'dan kişi track'leri  (ÖNCE tracker, sonra find_static!)
        tracks = yolo_tracker.update(frame)

        tracks_dict = {}
        person_track_ids = set()
        for tid, bbox, conf in tracks:
            tracks_dict[tid] = bbox
            person_track_ids.add(tid)
            track_last_seen[tid] = frame_idx

        # 4) Person mask (hem evaluate, hem de find_static için)
        person_mask = np.zeros((H, W), dtype=np.uint8)
        for tid, bbox, conf in tracks:
            x, y, w, h = clip_bbox(bbox, W, H)
            try:
                person_mask[y:y + h, x:x + w] = 255
            except Exception:
                pass

        # 5) Statik kandidatların tespiti
        #    BURADA person_mask'i DualFG'ye veriyoruz → insan bölgeleri statik blob olmayacak
        dets_static, labels_s, static_candidate = dual.find_static(
            mask_short,
            mask_long,
            roi_mask=roi_mask_uint8,
            person_mask=person_mask,    # <-- kritik ekleme
        )

        # 6) Moving components (şu an sadece debug/ileride kullanım için)
        nm, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(mask_short, connectivity=8)
        dets_move_raw = []
        for i in range(1, nm):
            x, y, w, h, area = stats_m[i]
            if area < 50:
                continue
            dets_move_raw.append([int(x), int(y), int(w), int(h), 1.0])
        keep = nms_boxes([d[:4] for d in dets_move_raw], None, iou_thresh=args.nms_iou)
        dets_move = [dets_move_raw[i] for i in keep]  # şu anda kullanılmıyor ama dursun

        # 7) Abandoned blob map güncelle
        abandoned.match_create(
            dets_static,
            labels_s,
            dual.static_counter,
            tracks_dict,
            person_track_ids,
            frame_idx,
        )

        # 8) Blob durumlarını değerlendir (candidate / abandoned / collected)
        evaluated = abandoned.evaluate(frame, dual.static_counter, person_mask, track_last_seen, frame_idx)

        # 9) overlay ve yeni protected mask
        new_prot_global = np.zeros((H, W), dtype=np.uint8)
        overlay = frame_proc.copy()

        for bid, bbox, status, conf, prot_mask in evaluated:
            x, y, w, h = bbox
            mask_uint8 = (prot_mask.astype(np.uint8) * 255).astype(np.uint8)

            if status == "abandoned":
                new_prot_global = np.logical_or(new_prot_global, prot_mask)
                overlay = cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                overlay = overlay_colored_mask(overlay, mask_uint8, (0, 0, 255), alpha=0.45)
                cv2.putText(
                    overlay,
                    f"ABND#{bid} {conf:.2f}",
                    (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
            else:
                overlay = cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 200), 1)
                overlay = overlay_colored_mask(overlay, mask_uint8, (0, 200, 200), alpha=0.25)
                cv2.putText(
                    overlay,
                    f"cand#{bid} {conf:.2f}",
                    (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )

        # 10) Long BG update (abandoned protected bölgeleri hariç)
        new_prot_uint8 = (new_prot_global.astype(np.uint8) * 255).astype(np.uint8)
        dual.update_long(frame, protected_mask=new_prot_uint8)

        # 11) Track kutuları çiz
        for tid, bbox, conf in tracks:
            x, y, w, h = clip_bbox(bbox, W, H)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"P#{tid}",
                (x, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        # Küçük debug mask görselleri (opsiyonel)
        try:
            sh = 120
            ms = cv2.resize(mask_short, (sh, sh))
            ml = cv2.resize(mask_long, (sh, sh))
            sc = cv2.resize(static_candidate, (sh, sh))
            top = np.hstack(
                [
                    cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(ml, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(sc, cv2.COLOR_GRAY2BGR),
                ]
            )
            overlay[0 : top.shape[0], 0 : top.shape[1]] = cv2.addWeighted(
                overlay[0 : top.shape[0], 0 : top.shape[1]], 0.6, top, 0.4, 0
            )
        except Exception:
            pass

        outv.write(overlay)

        if args.display:
            cv2.imshow("DualFG + YOLO BoT-SORT Abandoned", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - start_time
    print(f"[INFO] Done frames {frame_idx} time {elapsed:.2f}s fps {frame_idx / elapsed:.2f}")
    cap.release()
    outv.release()
    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
