- Create an empty Python project using Poetry:
  1. Install Poetry if you haven't already: `pip install poetry`
  2. Create a new project: `poetry new markdown_organizer`
  3. Navigate to the project directory: `cd markdown_organizer`
  4. Add Streamlit as a dependency: `poetry add streamlit`

- Create a Streamlit application to take markdown data and save them in organized folders as per input:
  1. Inside the `markdown_organizer` directory, create a new file named `app.py`.
  2. Implement the Streamlit application in `app.py` to take markdown data and save them in organized folders:
     - Import necessary libraries: `streamlit`, `os`, and `shutil`.
     - Create a Streamlit form to input markdown data and folder structure.
     - Save the markdown data in the specified folder structure.