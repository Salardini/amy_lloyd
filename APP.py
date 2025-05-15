# app.py
import os
import sys
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename  # For secure file uploads

# --- UTF-8 Setup (if needed for Flask's environment, though Flask handles Unicode well) ---
# This might not be strictly necessary here as Flask/browsers handle UTF-8 well
# but keeping it for consistency if other parts of your system expect it.
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# --- End UTF-8 Setup ---

# Import your existing logic. Assume they are in the same directory or PYTHONPATH
from input_parser import process_input_section  # We'll adapt this slightly
from text_extractor import extract_text_from_txt, extract_text_from_pdf, transcribe_audio_gcp
from deidentifier import basic_deidentify_text
import note_processing_core as core  # Your main note generation logic

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'  # Create this folder in your project directory
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'wav', 'mp3', 'm4a'}  # Add more as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit (adjust as needed)

# --- GCP Configuration (move to a config file or env variables for production) ---
GCP_PROJECT_ID_FOR_SPEECH = core.PROJECT_ID  # Assuming it's the same as for Vertex AI


# ----------------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_uploaded_files_and_text(form_data, files_data, section_name_prefix, deidentify_flag):
    """
    Processes direct text input and uploaded files for a given section.
    `form_data`: request.form from Flask
    `files_data`: request.files from Flask
    `section_name_prefix`: e.g., "background"
    `deidentify_flag`: boolean
    """
    section_texts = []

    # Process direct text input
    direct_text = form_data.get(f'{section_name_prefix}_text')
    if direct_text:
        processed_text = basic_deidentify_text(direct_text) if deidentify_flag else direct_text
        section_texts.append(processed_text)
        print(f"DEBUG: Added direct text for {section_name_prefix}")

    # Process uploaded files
    # In HTML, file inputs might be named like background_files_1, background_files_2, etc.
    # Or use a multiple file input: <input type="file" name="background_files" multiple>
    # For simplicity, let's assume a single file input per category for now,
    # or adapt to handle `request.files.getlist(f"{section_name_prefix}_files")` for multiple.

    file_key = f'{section_name_prefix}_file'  # Assuming single file input named like 'background_file'
    if file_key in files_data:
        file = files_data[file_key]
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_file_path)
            print(f"DEBUG: Saved temporary file {temp_file_path}")

            extracted_text = None
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'txt':
                extracted_text = extract_text_from_txt(temp_file_path)
            elif ext == 'pdf':
                extracted_text = extract_text_from_pdf(temp_file_path)
            elif ext in {'wav', 'mp3', 'm4a'}:  # Add other audio extensions
                if GCP_PROJECT_ID_FOR_SPEECH:
                    extracted_text = transcribe_audio_gcp(temp_file_path, GCP_PROJECT_ID_FOR_SPEECH)
                else:
                    print("ERROR: GCP_PROJECT_ID_FOR_SPEECH not configured for audio transcription.")

            if extracted_text:
                processed_text = basic_deidentify_text(extracted_text) if deidentify_flag else extracted_text
                section_texts.append(processed_text)
                print(f"DEBUG: Extracted and processed text from {filename}")

            # Clean up the temporary file
            try:
                os.remove(temp_file_path)
                print(f"DEBUG: Removed temporary file {temp_file_path}")
            except OSError as e:
                print(f"Error deleting temporary file {temp_file_path}: {e}")
        elif file and file.filename != '':
            print(f"WARNING: File type not allowed for {file.filename}")

    return "\n\n---\n\n".join(filter(None, section_texts))


@app.route('/')
def index():
    # Create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    return render_template('index.html')  # We'll create this HTML file


@app.route('/generate_note', methods=['POST'])
def generate_note_route():
    try:
        print("DEBUG: /generate_note endpoint hit")
        deidentify = request.form.get('deidentify') == 'on'  # Checkbox value
        print(f"DEBUG: De-identify flag: {deidentify}")

        background_info = process_uploaded_files_and_text(request.form, request.files, "background", deidentify)
        additional_info = process_uploaded_files_and_text(request.form, request.files, "additional", deidentify)
        transcription = process_uploaded_files_and_text(request.form, request.files, "transcription", deidentify)
        revised_info = process_uploaded_files_and_text(request.form, request.files, "revised", deidentify)

        print("\n--- Summary of Processed Inputs for Core Logic (first 100 chars) ---")
        print(f"Background Info: {background_info[:100]}...")
        print(f"Additional Info: {additional_info[:100]}...")
        print(f"Transcription: {transcription[:100]}...")
        print(f"Revised Info: {revised_info[:100]}...")
        print("----------------------------------------------------")

        # Call your core note generation logic
        generated_note = core.generate_full_note(
            background_info_text=background_info,
            additional_info_text=additional_info,
            transcription_text=transcription,
            revised_info_text=revised_info
        )

        if "ERROR:" in generated_note:
            return jsonify({'error': generated_note}), 500

        return jsonify({'note': generated_note})

    except Exception as e:
        print(f"ERROR in /generate_note: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    # Ensure core Vertex AI initialization happens when generate_full_note is called
    # Or initialize it once here if preferred, but core.py also does it.
    # core.initialize_vertex_ai() # Optional: initialize once at app start
    app.run(debug=True)  # debug=True is for development only