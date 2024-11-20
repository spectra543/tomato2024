import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # Import for centroid coordinates
from cv_bridge import CvBridge
import time


class DetectionPublisher(Node):
    def __init__(self, video_path):
        super().__init__('detection_publisher')
        self.detected_publisher_ = self.create_publisher(Image, 'detected_area', 10) # publishes every 0.1 second, to get 5 seconds: publish 50 messages
        self.original_publisher_ = self.create_publisher(Image, 'original_frame', 10)
        self.centroid_publisher_ = self.create_publisher(Point, 'object_centroid_diff', 10)
        self.all_mask_publisher_ = self.create_publisher(Image, 'all_detected_mask', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.model = YOLO("best_tomato2024.pt")
        self.names = self.model.model.names
        self.cap = cv2.VideoCapture(video_path)
        self.bridge = CvBridge()

        # Store the previous bounding box for "RTomato"
        self.previous_bounding_box = None
        self.blinking_threshold = 20  # Threshold for detecting blinking
        self.blinking_timespan = 5 * 10 # 5 seconds for 0.1 message/second makes 50 messages that need to be stored, it's the blinking timespan

    def timer_callback(self):
        ret, im0 = self.cap.read()
        if not ret:
            self.get_logger().info("End of video or failed to read frame.")
            rclpy.shutdown()
            return

        results = self.model.predict(im0)
        annotator = Annotator(im0, line_width=2)
        detected_area = np.zeros_like(im0)
        all_detected_mask = np.zeros_like(im0)

        # Frame centroid
        frame_height, frame_width = im0.shape[:2]
        frame_centroid_x, frame_centroid_y = frame_width // 2, frame_height // 2

        # Draw frame centroid on the original frame
        cv2.circle(im0, (frame_centroid_x, frame_centroid_y), 10, (200, 50, 150), -1)

        # Define the target classes you want to detect only one instance of each
        target_classes = {"GTomato", "RTomato", "HTomato", "stem"}
        detected_classes = set()

        if len(results[0].boxes) > 0:
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for idx in range(len(boxes)):
                x1, y1, x2, y2 = boxes[idx]
                cls = clss[idx]
                class_name = self.names[int(cls)]

                if class_name == "RTomato":
                    self.get_logger().info(f"Bounding box of red tomato at: x: {x1} - {x2}; y: {y1} - {y2}")

                    # Check if the bounding box is blinking
                    if self.previous_bounding_box:
                        px1, py1, px2, py2 = self.previous_bounding_box
                        if (abs(x1 - px1) > self.blinking_threshold or
                                abs(y1 - py1) > self.blinking_threshold or
                                abs(x2 - px2) > self.blinking_threshold or
                                abs(y2 - py2) > self.blinking_threshold):
                            self.get_logger().warn("BLINKING TOMATO")

                    # Update the previous bounding box
                    self.previous_bounding_box = (x1, y1, x2, y2)

                    # Calculate centroid of the bounding box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(im0, (cx, cy), 5, (255, 200, 100), -1)
                    self.get_logger().info(f"Centroid of {class_name}: ({cx}, {cy})")

                    # Calculate the difference between object and frame centroids
                    diff_x = cx - frame_centroid_x
                    diff_y = cy - frame_centroid_y

                    # Publish the centroid difference
                    self.publish_centroid_difference(diff_x, diff_y)

        # Publish the detected area with bounding boxes
        self.publish_detected_area(detected_area)
        # Publish the original frame with the red dot
        self.publish_original_frame(im0)
        # Publish the mask with all detected objects
        self.publish_all_detected_mask(all_detected_mask)

    def publish_detected_area(self, detected_area):
        ros_image = self.bridge.cv2_to_imgmsg(detected_area, encoding="bgr8")
        self.detected_publisher_.publish(ros_image)
        self.get_logger().info("Published detected area.")

    def publish_original_frame(self, original_frame):
        ros_image = self.bridge.cv2_to_imgmsg(original_frame, encoding="bgr8")
        self.original_publisher_.publish(ros_image)
        self.get_logger().info("Published original frame.")
        
    def publish_all_detected_mask(self, all_detected_mask):
        ros_image = self.bridge.cv2_to_imgmsg(all_detected_mask, encoding="bgr8")
        self.all_mask_publisher_.publish(ros_image)
        self.get_logger().info("Published all detected mask.")

    def publish_centroid_difference(self, diff_x, diff_y):
        point_msg = Point()
        point_msg.x = float(diff_x)
        point_msg.y = float(diff_y)
        point_msg.z = 0.0
        self.centroid_publisher_.publish(point_msg)
        self.get_logger().info(f"Published centroid difference: x={diff_x}, y={diff_y}")


def main(args=None):
    rclpy.init(args=args)
    video_path = './videos/IMG_1060.mov'  # Update this with your video file path
    detection_publisher = DetectionPublisher(video_path)
    rclpy.spin(detection_publisher)
    detection_publisher.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
