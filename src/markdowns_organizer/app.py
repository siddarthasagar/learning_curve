from pathlib import Path

import streamlit as st


def save_markdown(data: str, folder: str) -> None:
    """Save markdown data to a specified folder."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "data.md"
    file_path.write_text(data)


st.title("Markdown Organizer")

markdown_data = st.text_area("Enter your markdown data here:")
folder_name = st.text_input("Enter the folder name to save the markdown data:")

if st.button("Save"):
    save_markdown(markdown_data, folder_name)
    st.success(f"Markdown data saved in folder: {folder_name}")
