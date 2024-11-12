import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SegmentationPublisher(Node):
    def __init__(self):
        super().__init__('segmentation_publisher')
        self.segmented_publisher_ = self.create_publisher(Image, 'segmented_area', 10)
        self.original_publisher_ = self.create_publisher(Image, 'original_frame', 10)  # Publisher for original frames
        self.timer = self.create_timer(0.1, self.timer_callback)  # Adjust the frequency as needed
        self.model = YOLO("best_v8.pt")  # Load the segmentation model
        self.names = self.model.model.names  # Get the class names
        self.cap = cv2.VideoCapture("testAll.mp4")
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, im0 = self.cap.read()
        if not ret:
            self.get_logger().info("Video frame is empty or video processing has been successfully completed.")
            return

        results = self.model.predict(im0)
        annotator = Annotator(im0, line_width=2)
        dark_background = np.zeros_like(im0)

        # frame centroid
        frame_height, frame_width = im0.shape[:2]
        frame_centroid_x, frame_centroid_y = frame_width//2, frame_height//2

        # map centroid
        cv2.circle(im0, (frame_centroid_x, frame_centroid_y), 10, (200, 50, 150), -1)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy

            # Calculate the area of each mask and store it in a list
            mask_areas = [cv2.contourArea(np.array(mask, dtype=np.int32)) for mask in masks]

            # Sort the indices based on mask areas in descending order
            sorted_indices = sorted(range(len(mask_areas)), key=lambda i: mask_areas[i], reverse=True)

            # Use only the object with the largest mask area
            if sorted_indices:
                idx = sorted_indices[0]  # Get index of the object with the largest mask area
                mask = masks[idx]
                cls = clss[idx]

                if mask.size > 0:  # Check if mask has valid data
                    color = colors(int(cls), True)
                    txt_color = annotator.get_txt_color(color)

                    # Create a mask for the segmented area
                    mask_img = np.zeros(im0.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_img, [np.array(mask, dtype=np.int32)], 255)
                    highlighted_area = cv2.bitwise_and(im0, im0, mask=mask_img)
                    dark_background = cv2.add(dark_background, highlighted_area)

                    # rectangle draw
                    x, y, w, h = cv2.boundingRect(np.array(mask, dtype=np.int32))
                    cv2.rectangle(dark_background, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(im0, (x, y), (x + w, y + h), color, 2)

                    # Calculate moments to find the centroid
                    M = cv2.moments(mask_img)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Draw centroid
                        cv2.circle(dark_background, (cx, cy), 5, (0, 255, 0), -1)
                        self.get_logger().info(f"Centroid of the largest mask: ({cx}, {cy})")

                    # Optionally, add bounding box and label for the single detected tomato
                    annotator.seg_bbox(mask=mask, mask_color=color, label=self.names[int(cls)], txt_color=txt_color)

        # Publish the highlighted segmented area
        self.publish_segmented_area(dark_background)
        # Publish the original frame
        self.publish_original_frame(im0)

    def publish_segmented_area(self, segmented_area):
        # Convert the segmented area to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(segmented_area, encoding="bgr8")
        self.segmented_publisher_.publish(ros_image)
        self.get_logger().info("Published segmented area.")

    def publish_original_frame(self, original_frame):
        # Convert the original frame to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(original_frame, encoding="bgr8")
        self.original_publisher_.publish(ros_image)
        self.get_logger().info("Published original frame.")


def main(args=None):
    rclpy.init(args=args)
    segmentation_publisher = SegmentationPublisher()
    rclpy.spin(segmentation_publisher)
    segmentation_publisher.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
