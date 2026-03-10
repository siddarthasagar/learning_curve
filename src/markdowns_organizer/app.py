import streamlit as st
import os

# Function to save markdown data to a specified folder
def save_markdown(data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, "data.md")
    with open(file_path, "w") as file:
        file.write(data)

st.title("Markdown Organizer")

# Input for markdown data
markdown_data = st.text_area("Enter your markdown data here:")

# Input for folder name
folder_name = st.text_input("Enter the folder name to save the markdown data:")

if st.button("Save"):
    save_markdown(markdown_data, folder_name)
    st.success(f"Markdown data saved in folder: {folder_name}")
