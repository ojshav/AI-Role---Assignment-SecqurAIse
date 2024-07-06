import cv2
import numpy as np

def process_video(input_video_path, output_video_path, output_text_file):
    # Define the color ranges for the balls with refined ranges to exclude background
    color_ranges = {
        'red': ([0, 120, 70], [10, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'green': ([35, 50, 50], [85, 255, 255])  # More precise green range
    }
    
    # Initialize video capture and get video properties
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Define quadrants (assuming 4 quadrants for simplicity)
    quadrants = {
        1: (0, 0, width // 2, height // 2),
        2: (width // 2, 0, width, height // 2),
        3: (0, height // 2, width // 2, height),
        4: (width // 2, height // 2, width, height)
    }
    
    # Initialize data structures for tracking
    ball_positions = {color: None for color in color_ranges.keys()}
    events = []
    
    # Helper function to check which quadrant a point is in
    def get_quadrant(x, y):
        for quadrant, (x1, y1, x2, y2) in quadrants.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return quadrant
        return None
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 10:
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    quadrant = get_quadrant(center[0], center[1])
                    if ball_positions[color] != quadrant:
                        if ball_positions[color] is not None:
                            events.append((frame_count / fps, ball_positions[color], color, "Exit"))
                            cv2.putText(frame, f"Exit {ball_positions[color]} {color}", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        ball_positions[color] = quadrant
                        events.append((frame_count / fps, quadrant, color, "Entry"))
                        cv2.putText(frame, f"Entry {quadrant} {color}", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(center[0] - radius), int(center[1] - radius)), 
                                  (int(center[0] + radius), int(center[1] + radius)), (0, 255, 255), 2)
                    cv2.putText(frame, color, (int(center[0] - radius), int(center[1] - radius) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    with open(output_text_file, 'w') as f:
        for event in events:
            f.write(f"{event[0]:.2f}, {event[1]}, {event[2]}, {event[3]}\n")

# Define input and output paths directly in the script
input_video_path = 'AI Assignment video.mp4'
output_video_path = 'output_video.avi'
output_text_file = 'output_events.txt'

process_video(input_video_path, output_video_path, output_text_file)
