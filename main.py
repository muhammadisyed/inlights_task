import cv2
import os
import sys
from collections import defaultdict
from utils import initialize_model, process_stream, draw_boxes, click_event

class ObjectTracker:
    def __init__(self, rtsp_link, model_path='yolov8n.pt'):
        self.rtsp_link = rtsp_link
        self.model = initialize_model(model_path)
        self.track_history = defaultdict(lambda: [])
        self.colors = {}
        self.boxes = []
        self.selected_box = None
        self.start_time = None

    def run(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        try:
            stream = cv2.VideoCapture(self.rtsp_link)
        except Exception as e:
            print("Error: Could not open stream")
            return

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", click_event, self)

        while True:
            ret, img_res = stream.read()
            if ret:
                results, self.boxes, track_ids, class_names = process_stream(self.model, img_res)
                frame_with_boxes = draw_boxes(img_res.copy(), self.boxes, track_ids, class_names, self.colors, self.selected_box, self.start_time, self.track_history)

                cv2.imshow("Frame", frame_with_boxes)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <rtsp_link>")
        return

    rtsp_link = sys.argv[1]
    tracker = ObjectTracker(rtsp_link)
    tracker.run()

if __name__ == "__main__":
    main()
