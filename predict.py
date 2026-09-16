from ultralytics import YOLO
import supervision as sv

def count_objects(list_of_labels):
    object_counts = {}
    for label in list_of_labels:
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1
    return object_counts

def predict(image, model_path):
    # model = YOLO("runs/detect/train/weights/best.pt")  # Load the YOLOv12n model
    model = YOLO(model_path)  # Load the specified YOLO model
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)

    predicted_labels = [results.names[class_id] for class_id in detections.class_id]

    return count_objects(predicted_labels), annotated_image