import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd  # For loading and handling the CSV

# Load the trained YOLOv8 model
model = YOLO('best__3_.pt') 

# Load the nutritional data CSV file
# Ensure the CSV has the following columns: 'Food Item', 'Calories', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)'
nutrition_df = pd.read_csv('Nutritional Breakdown - Sheet1.csv')  # Update with your actual CSV filename

# Normalize the 'Food Item' column to lowercase and strip any extra whitespace
nutrition_df['Food Item'] = nutrition_df['Food Item'].str.lower().str.strip()

# Streamlit app title
st.title("Food Item Detection and Nutritional Breakdown App")

# User can choose between file upload or camera input
st.write("You can either upload an image or scan using your camera.")

# File uploader option
uploaded_image = st.file_uploader("Upload an image of a food item", type=["jpg", "png", "jpeg"])

# Camera input option
captured_image = st.camera_input("Or, take a photo using your camera")

# Process the image if either a file is uploaded or a photo is taken
if uploaded_image or captured_image:
    # Reset session state when a new image is uploaded or captured
    if 'item_counts' in st.session_state:
        del st.session_state['item_counts']  # Clear previous results

    # If an image was uploaded, use that, otherwise use the captured image from the camera
    image = Image.open(uploaded_image) if uploaded_image else Image.open(captured_image)

    # Display the image on Streamlit
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Perform food detection using the YOLO model
    st.write("Detecting food items...")

    # Run YOLOv8 inference on the uploaded/captured image
    results = model(image, imgsz=640, conf=0.3, iou=0.7)

    # Extract detection results
    boxes = results[0].boxes

    # Dictionary to store detected items and their counts
    item_counts = {}

    for box in boxes:
        # Get class name and confidence
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Normalize the class name for comparison with the CSV
        class_name_normalized = class_name.lower().strip()

        # Count the number of detected instances for each food item
        if class_name_normalized in item_counts:
            item_counts[class_name_normalized] += 1
        else:
            item_counts[class_name_normalized] = 1

    # Store the counts of food items in the session state
    st.session_state['item_counts'] = item_counts

    # Allow the user to adjust the counts of detected items
    if st.session_state['item_counts']:
        st.write("Detected Items and Nutritional Breakdown:")

        for item_name, count in st.session_state['item_counts'].items():
            # Lookup nutritional information for the detected item
            nutrition_info = nutrition_df[nutrition_df['Food Item'] == item_name]

            if not nutrition_info.empty:
                # Allow user to adjust the count of each food item
                new_count = st.number_input(f"{item_name.capitalize()} (Detected {count} times)", min_value=0,
                                            value=count)

                # Update the item count based on user input
                st.session_state['item_counts'][item_name] = new_count

                # Multiply the nutritional values by the count of detected items
                total_calories = nutrition_info['Calories'].values[0] * new_count
                total_protein = nutrition_info['Protein (g)'].values[0] * new_count
                total_carbohydrates = nutrition_info['Carbohydrates (g)'].values[0] * new_count
                total_fat = nutrition_info['Fat (g)'].values[0] * new_count
                total_fiber = nutrition_info['Fiber (g)'].values[0] * new_count

                # Display the nutritional breakdown multiplied by the count
                st.write(f"**{item_name.capitalize()}**")
                st.write(f"Nutritional Breakdown (for {new_count} item{'s' if new_count > 1 else ''}):")
                st.write(f"Calories: {total_calories}")
                st.write(f"Protein: {total_protein}g")
                st.write(f"Carbohydrates: {total_carbohydrates}g")
                st.write(f"Fat: {total_fat}g")
                st.write(f"Fiber: {total_fiber}g")
            else:
                st.write(f"Nutritional information for {item_name.capitalize()} is not available.")
    else:
        st.write("No food items detected in the image.")

    # Display the image with bounding boxes
    st.image(results[0].plot(), caption="Detected Food Items", use_column_width=True)
