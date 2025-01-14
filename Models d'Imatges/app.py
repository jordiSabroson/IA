import cv2
import time
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Àrea delimitada on volem que es detectin els elements (he agafat un frame del video i amb
# makesense.ai he fet l'àrea). Es un array d'arrays on el primer element es la x i l'altra es la y 
POLYGON = np.array([
    [0.4601693075754246,196.87938644092316],
    [71.43549670790823,172.19231604080738],
    [168.33224802836256,154.91136676072637],
    [261.5259387887995,151.8254829607119],
    [334.352796469141,157.38007380073793],
    [404.7109471094709,143.1850083206714],
    [479.9999999999999,172.8094928008103],
    [479.9999999999999,665.3165472831196],
    [394.21894218942174,688.7692641632295],
    [334.352796469141,696.7925620432673],
    [202.27696982852174,696.7925620432673],
    [112.16916286809922,685.0662036032122],
    [56.00607770783588,668.402431083134],
    [0,646.8012444830329]
], dtype=np.int32)


model = YOLO("yolo11n.pt") # Importem el model YOLO 11 nano d'ultralytics
# print(model.names) per veure els elements que pot detectar Yolo

polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))

box_annotator = sv.BoxAnnotator() # Annotator que farem servir per resaltar els elements detectats

def main(video_file_path):
    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)

    # Definir resolució desitjada
    target_width = 480
    aspect_ratio = None

    for frame in frame_generator:

        # Calcular aspect ratio només una vegada
        if aspect_ratio is None:
            original_height, original_width = frame.shape[:2]
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)

        resized_frame = cv2.resize(frame, (target_width, target_height)) # Redimensionar el frame
        
        result = model(resized_frame, device="cuda", verbose=False, imgsz=1280)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)] #Fem que les deteccions estiguin dins del poligon
        detections = detections[detections.class_id == 0] # Amb el model.names veiem que l'id 0 són les persones

        annotated_frame = resized_frame.copy() # Còpia del frame original per anotar la còpia i no l'original

        # Característiques del polígon de la delimitació
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=POLYGON,
            color=sv.Color.RED,
            thickness=2
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        time.sleep(0.5)
        cv2.imshow("Processed video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file_path")

    args = parser.parse_args()

    main(args.video_file_path)