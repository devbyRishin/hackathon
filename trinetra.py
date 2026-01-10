import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client
import time
from IPython.display import display, Image as IPImage, clear_output


try:
    client_twilio = Client(TWILIO_SID, TWILIO_AUTH)
except Exception as e:
    print(f"Twilio Setup Error: {e}")

alert_cooldown = 90  
last_alert_time = 0
SKELETON = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
            (11,12), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16)]

model = YOLO("yolov8n-pose.pt")
url = "http://192.168.193.159:8085/?action=stream"
cap = cv2.VideoCapture(url)

def send_emergency_alerts(count, lat, lon):
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > alert_cooldown:
        try:
            msg = f"EMERGENCY: {count} SURVIVORS detected at {lat}N, {lon}E"
            client_twilio.messages.create(body=msg, from_=TWILIO_PHONE, to=TARGET_PHONE)
            
            client_twilio.calls.create(
                twiml=f'<Response><Say>Emergency! {count} survivors detected.</Say></Response>',
                from_=TWILIO_PHONE, to=TARGET_PHONE
            )
            last_alert_time = current_time
            print(f"Alerts sent for {count} survivors.")
        except Exception as e:
            print(f"Twilio failed: {e}")

print("System Active. Press 'q' to exit.")

try:
    frame_count = 0
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame_count += 1
        frame = cv2.resize(cv2.flip(frame, -1), (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tactical Visuals
        normal_view = frame.copy()
        
        night_vision = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        night_vision[:, :, 0] = 0  
        night_vision[:, :, 2] = 0  
        
        thermal_view = cv2.applyColorMap(cv2.bitwise_not(gray), cv2.COLORMAP_INFERNO)

        if frame_count % 3 == 0:
            results = model(frame, conf=0.45, verbose=False)
            last_results = results[0] if results else None

        survivor_count = 0
        if last_results and last_results.keypoints is not None:
            survivor_count = len(last_results.boxes)
            
            if survivor_count > 0:
                send_emergency_alerts(survivor_count, 20.9467, 72.9520)

            for i, box in enumerate(last_results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                roi = gray[y1:y2, x1:x2]
                sim_temp = 36.0 + (np.mean(roi) / 255.0) * 1.8 if roi.size > 0 else 0
                
                for view in [normal_view, night_vision, thermal_view]:
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    if view is thermal_view:
                        cv2.putText(view, f"{sim_temp:.1f}C", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    kpts = last_results.keypoints.xy[i].cpu().numpy()
                    for start, end in SKELETON:
                        if kpts[start][0] > 0 and kpts[end][0] > 0:
                            p1 = (int(kpts[start][0]), int(kpts[start][1]))
                            p2 = (int(kpts[end][0]), int(kpts[end][1]))
                            cv2.line(view, p1, p2, (0, 255, 0), 1)

        # HUD Assembly
        res_size = (400, 300)
        combined = np.hstack((
            cv2.resize(normal_view, res_size), 
            cv2.resize(night_vision, res_size), 
            cv2.resize(thermal_view, res_size)
        ))

        cv2.rectangle(combined, (0, 0), (1200, 40), (20, 20, 20), -1)
        hud_label = f"SAR SCAN | SURVIVORS: {survivor_count} | LOC: 34.9467 N, 118.9520 E"
        cv2.putText(combined, hud_label, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Tactical SAR", combined)
        
        _, encoded_img = cv2.imencode(".jpg", combined)
        display(IPImage(data=encoded_img))
        clear_output(wait=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Manual shutdown.")
finally:
    cap.release()
    cv2.destroyAllWindows()
