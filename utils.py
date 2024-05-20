import cv2
import random
import time
from ultralytics import YOLO

def initialize_model(model_path):
    return YOLO(model_path)

def draw_boxes(frame, boxes, track_ids, class_names, colors, selected_box, start_time, track_history):
    for i, (box, track_id, class_name) in enumerate(zip(boxes, track_ids, class_names)):
        if track_id not in colors:
            colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color = colors[track_id]
        label = f'{class_name} ID: {track_id}'
        if selected_box is not None and i == selected_box:
            color = (0, 0, 255)
            if start_time is not None:
                elapsed_time = int(time.time() - start_time)
                label = f'{class_name} ID: {track_id} Time: {elapsed_time}s'
        
        x, y, w, h = map(int, box)
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 90:
            track.pop(0)

    return frame

def click_event(event, x, y, flags, param):
    tracker = param
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(tracker.boxes):
            x_center, y_center, w, h = map(int, box)
            x1, y1, x2, y2 = x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2
            if x1 <= x <= x2 and y1 <= y <= y2:
                tracker.selected_box = i
                tracker.start_time = time.time()
                break

def process_stream(model, img_res):
    results = model.track(img_res, persist=True)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_names = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist()]
    else:
        boxes = []
        track_ids = []
        class_names = []
    return results, boxes, track_ids, class_names
