import streamlit as st
import predict
import PIL

st.set_page_config(page_title="Vehicle Detection with YOLOv12n", page_icon="🤖")

# streamlit UI setup
st.title("🤖 Vehicle Detection with YOLOv12n")
st.caption("Detect and classify vehicles in real-time using advanced computer vision")
st.divider()

st.caption("Upload an image to detect vehicles and classify them using the YOLOv12n model.")
st.caption("3 detectable classes: Car, Truck, Van")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    image = PIL.Image.open(uploaded_file)
    # Perform prediction and count objects
    object_counts, annotated_image = predict.predict(image)

    # Display the annotated image
    st.subheader("Annotated Image")
    st.image(annotated_image, caption="Annotated Image", width='stretch')

    # Display the object counts
    st.subheader("Object Counts")
    for label, count in object_counts.items():
        st.markdown(f"* {label.upper()}: {count}")