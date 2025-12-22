cat > fabric_defect_opencv.py << 'ENDOFSCRIPT'
#!/usr/bin/env python3
"""
Real-Time Fabric Hole Detection - OPTIMIZED
For Jetson Orin Nano with USB Camera
Strict detection - only real holes, not color variations
"""

import cv2
import numpy as np
import time
from collections import deque

class FabricHoleDetector:
    def __init__(self, camera_id=0):
        """Initialize the detector with camera"""
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        
        # STRICT hole detection parameters
        self.min_hole_area = 300       # Bigger minimum - ignore tiny specks
        self.max_hole_area = 20000     # Maximum hole size
        self.color_tolerance = 50      # Higher = more lenient (START HIGH)
        self.min_circularity = 0.3     # Holes should be somewhat round
        self.min_color_diff = 60       # Minimum color difference to consider
        
        # Fabric color learning
        self.fabric_color = None
        self.fabric_std = None         # Standard deviation - for filtering
        self.fabric_color_samples = []
        self.learning_mode = True
        self.samples_needed = 60
        
        # HSV ranges
        self.lower_bound = None
        self.upper_bound = None
        
        # Performance
        self.fps_buffer = deque(maxlen=30)
        
    def initialize_camera(self):
        """Setup the USB camera"""
        print("[INFO] Initializing camera...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable auto exposure and white balance for consistent colors
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Camera: {actual_width}x{actual_height}")
        print("[INFO] LEARNING MODE - Show good fabric (no holes) for 3 seconds")
        print("[INFO] Keep fabric steady and well-lit!")
        self.running = True
        
    def learn_fabric_color(self, frame):
        """Learn the fabric color with better sampling"""
        h, w = frame.shape[:2]
        
        # Sample from multiple regions, not just center
        regions = [
            (w//4, h//4, w//2, h//2),      # Center
            (w//6, h//6, w//3, h//3),      # Top-left area
            (2*w//3, h//6, w//3, h//3),    # Top-right area
            (w//6, 2*h//3, w//3, h//3),    # Bottom-left area
            (2*w//3, 2*h//3, w//3, h//3),  # Bottom-right area
        ]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sample_colors = []
        
        for x, y, rw, rh in regions:
            roi = hsv[y:y+rh, x:x+rw]
            if roi.size > 0:
                median_color = np.median(roi.reshape(-1, 3), axis=0)
                sample_colors.append(median_color)
                # Draw sampling areas
                cv2.rectangle(frame, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
        
        if len(sample_colors) > 0:
            frame_color = np.median(sample_colors, axis=0)
            self.fabric_color_samples.append(frame_color)
        
        samples_left = self.samples_needed - len(self.fabric_color_samples)
        cv2.putText(frame, f"Learning: {samples_left} frames left", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Keep fabric steady!", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Done learning
        if len(self.fabric_color_samples) >= self.samples_needed:
            samples_array = np.array(self.fabric_color_samples)
            self.fabric_color = np.median(samples_array, axis=0)
            self.fabric_std = np.std(samples_array, axis=0)  # Measure variability
            
            # Set STRICT color ranges based on learned color and its variation
            # Use 2x standard deviation for more robust filtering
            h_range = max(self.color_tolerance, 2 * self.fabric_std[0])
            s_range = max(40, 2 * self.fabric_std[1])
            v_range = max(40, 2 * self.fabric_std[2])
            
            self.lower_bound = np.array([
                max(0, self.fabric_color[0] - h_range),
                max(0, self.fabric_color[1] - s_range),
                max(0, self.fabric_color[2] - v_range)
            ])
            
            self.upper_bound = np.array([
                min(180, self.fabric_color[0] + h_range),
                min(255, self.fabric_color[1] + s_range),
                min(255, self.fabric_color[2] + v_range)
            ])
            
            self.learning_mode = False
            print(f"\n[INFO] Fabric color learned!")
            print(f"[INFO] HSV: H={self.fabric_color[0]:.0f}, S={self.fabric_color[1]:.0f}, V={self.fabric_color[2]:.0f}")
            print(f"[INFO] Variability: H±{self.fabric_std[0]:.1f}, S±{self.fabric_std[1]:.1f}, V±{self.fabric_std[2]:.1f}")
            print("[INFO] Starting hole detection...\n")
    
    def detect_holes(self, frame):
        """Detect holes with STRICT filtering"""
        if self.fabric_color is None:
            return [], None
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create fabric mask
        fabric_mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Invert to get potential holes
        hole_mask = cv2.bitwise_not(fabric_mask)
        
        # AGGRESSIVE noise removal
        # Remove small isolated pixels
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hole_mask = cv2.erode(hole_mask, kernel_erode, iterations=2)
        
        # Fill in the actual holes
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        hole_mask = cv2.dilate(hole_mask, kernel_dilate, iterations=2)
        
        # Remove anything touching the border (usually lighting issues)
        h, w = hole_mask.shape
        border_size = 20
        hole_mask[0:border_size, :] = 0
        hole_mask[h-border_size:h, :] = 0
        hole_mask[:, 0:border_size] = 0
        hole_mask[:, w-border_size:w] = 0
        
        # Find contours
        contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        holes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter
            if area < self.min_hole_area or area > self.max_hole_area:
                continue
            
            # Get properties
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filter - holes shouldn't be super elongated
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Circularity filter - holes should be somewhat round
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Color difference check - must be SIGNIFICANTLY different
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            hole_region_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
            hole_color = cv2.mean(hole_region_hsv, mask=mask)[:3]
            
            # Calculate color distance
            color_diff = np.sqrt(np.sum((np.array(hole_color) - self.fabric_color) ** 2))
            
            # Must have significant color difference
            if color_diff < self.min_color_diff:
                continue
            
            # Solidity check - real holes are solid, not scattered pixels
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.7:  # Must be at least 70% solid
                continue
            
            # Passed all filters - this is likely a real hole
            confidence = min(
                (color_diff / 100) * 0.5 +      # Color difference
                circularity * 0.3 +              # Roundness
                min(area / 1000, 1.0) * 0.2,     # Size
                1.0
            )
            
            holes.append({
                'bbox': (x, y, w, h),
                'area': area,
                'circularity': circularity,
                'color_diff': color_diff,
                'confidence': confidence,
                'solidity': solidity
            })
        
        # Sort by confidence
        holes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return holes, hole_mask
    
    def draw_holes(self, frame, holes):
        """Draw only high-confidence holes"""
        drawn_count = 0
        
        for hole in holes:
            # Only draw holes with decent confidence
            if hole['confidence'] < 0.4:
                continue
            
            x, y, w, h = hole['bbox']
            confidence = hole['confidence']
            
            # Color coding
            if confidence > 0.7:
                color = (0, 0, 255)      # Red - definitely a hole
                thickness = 4
            elif confidence > 0.55:
                color = (0, 100, 255)    # Orange - probably a hole
                thickness = 3
            else:
                color = (0, 200, 255)    # Light orange - check it
                thickness = 2
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw X
            cv2.line(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x + w, y), (x, y + h), color, 2)
            
            # Label with details
            text = f"HOLE {int(confidence * 100)}%"
            detail = f"Size:{int(hole['area'])} C:{hole['circularity']:.2f}"
            
            # Background for text
            label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 35), 
                         (x + max(label_size[0], 150), y), color, -1)
            
            # Text
            cv2.putText(frame, text, (x + 5, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, detail, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            drawn_count += 1
        
        return frame, drawn_count
    
    def add_overlay(self, frame, hole_count, fps):
        """Status overlay"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Status
        if self.learning_mode:
            status_text = "LEARNING FABRIC..."
            status_color = (0, 255, 255)
        elif hole_count == 0:
            status_text = "✓ NO HOLES DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = f"✗ {hole_count} HOLE(S) FOUND"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if not self.learning_mode:
            cv2.putText(frame, f"Min Size: {self.min_hole_area}px", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Color Diff: {self.min_color_diff}", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Circularity: {self.min_circularity:.1f}", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(frame, "Q:Quit | S:Save | R:Relearn | +:Size | -:Size | C:ColorDiff | V:Circularity", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        self.initialize_camera()
        
        print("\n" + "="*70)
        print("OPTIMIZED FABRIC HOLE DETECTION")
        print("="*70)
        print("Features:")
        print("  ✓ Strict size filtering (ignores tiny specks)")
        print("  ✓ Circularity check (holes are roundish)")
        print("  ✓ Color difference threshold (must be significant)")
        print("  ✓ Solidity check (filters scattered pixels)")
        print("  ✓ Border exclusion (ignores lighting edges)")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  R - Relearn fabric")
        print("  +/- - Adjust min size")
        print("  C - Cycle color difference threshold")
        print("  V - Cycle circularity threshold")
        print("="*70 + "\n")
        
        screenshot_count = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                output = frame.copy()
                
                if self.learning_mode:
                    self.learn_fabric_color(output)
                    holes = []
                    drawn_count = 0
                else:
                    holes, hole_mask = self.detect_holes(frame)
                    output, drawn_count = self.draw_holes(output, holes)
                    
                    # Show mask in corner
                    if hole_mask is not None:
                        small_mask = cv2.resize(hole_mask, (160, 120))
                        small_mask_colored = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
                        h_out, w_out = output.shape[:2]
                        output[h_out-130:h_out-10, w_out-170:w_out-10] = small_mask_colored
                        cv2.rectangle(output, (w_out-172, h_out-132), (w_out-8, h_out-8), (255, 255, 255), 2)
                        cv2.putText(output, "Detection", (w_out-165, h_out-138),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
                
                # Overlay
                output = self.add_overlay(output, drawn_count, avg_fps)
                
                cv2.imshow('Fabric Hole Detection - OPTIMIZED', output)
                
                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    filename = f"hole_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"[SAVED] {filename}")
                    screenshot_count += 1
                elif key == ord('r'):
                    self.fabric_color = None
                    self.fabric_color_samples = []
                    self.learning_mode = True
                    print("[INFO] Relearning fabric color...")
                elif key == ord('+') or key == ord('='):
                    self.min_hole_area = max(100, self.min_hole_area - 50)
                    print(f"[INFO] Min hole size: {self.min_hole_area}px")
                elif key == ord('-') or key == ord('_'):
                    self.min_hole_area = min(2000, self.min_hole_area + 50)
                    print(f"[INFO] Min hole size: {self.min_hole_area}px")
                elif key == ord('c'):
                    self.min_color_diff += 10
                    if self.min_color_diff > 100:
                        self.min_color_diff = 30
                    print(f"[INFO] Color difference threshold: {self.min_color_diff}")
                elif key == ord('v'):
                    self.min_circularity += 0.1
                    if self.min_circularity > 0.8:
                        self.min_circularity = 0.2
                    print(f"[INFO] Circularity threshold: {self.min_circularity:.1f}")
                
        except KeyboardInterrupt:
            print("\n[INFO] Stopped")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done!")

def main():
    detector = FabricHoleDetector(camera_id=0)
    try:
        detector.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        detector.cleanup()

if __name__ == "__main__":
    main()
ENDOFSCRIPT

python3 fabric_defect_opencv.py