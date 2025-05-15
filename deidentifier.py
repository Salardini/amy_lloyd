# deidentifier.py
import subprocess
import tempfile
import os
import re  # For basic de-identifier
import sys  # For sys.stdout.encoding

# --- NLM Scrubber Configuration ---
# !! IMPORTANT !!
# 1. Ensure Scrubber.exe is accessible via system PATH or provide the full path.
# 2. Update SCRUBBER_EXE_PATH to the correct absolute path if not in PATH.
# 3. You WILL LIKELY NEED TO ADJUST THE COMMAND-LINE ARGUMENTS FOR SCRUBBER.
#    Consult the documentation for the version of Scrubber you are using.
#    The example below uses hypothetical flags.

SCRUBBER_EXE_PATH = "C:\Users\u2121\projects\neuroapp\scrubber.19.0411W\scrubber.19.0411W.exe"  # Assumes scrubber.exe is in PATH or current dir.
# OR "C:/path/to/your/scrubber.exe"
# OR "./tools/scrubber.exe" (relative path)

# Example Scrubber options (THESE ARE HYPOTHETICAL - CHECK SCRUBBER DOCS):
SCRUBBER_DEFAULT_OPTIONS = [
    # "-config", "path/to/scrubber_phi_config.xml", # Example config for what to scrub
    # "-phi", "ALL", # Example: scrub all PHI types it knows
    # "-silent", # Example if it has a silent mode
]


# ----------------------------------

def deidentify_text_with_scrubber(text_content):
    if not text_content:
        return ""

    # Check if the executable path seems valid (simple check)
    # For a more robust check, you might use shutil.which(SCRUBBER_EXE_PATH)
    # but that only works if it's in PATH.
    if "\\" not in SCRUBBER_EXE_PATH and "/" not in SCRUBBER_EXE_PATH:  # Likely just "scrubber.exe"
        # Try to see if it's in PATH, otherwise this might fail later
        pass
    elif not os.path.exists(SCRUBBER_EXE_PATH) and not os.path.isfile(SCRUBBER_EXE_PATH):
        print(f"ERROR: NLM Scrubber EXE not found at '{SCRUBBER_EXE_PATH}'. Scrubber will not be used.")
        return f"[SCRUBBER_ERROR: EXE NOT FOUND AT SPECIFIED PATH] {text_content}"

    fd_in, infile_path = tempfile.mkstemp(suffix=".txt", text=True, encoding="utf-8")
    # We need to close the file descriptor from mkstemp before Scrubber tries to use the output path
    # So, create the name, then close fd, then let Scrubber write, then open for read.
    out_fd_handle, outfile_path = tempfile.mkstemp(suffix=".txt", text=True, encoding="utf-8")
    os.close(out_fd_handle)  # Close the handle, we just need the path for Scrubber to write to

    try:
        with os.fdopen(fd_in, "w", encoding="utf-8") as f_write:
            f_write.write(text_content)

        # Command for Scrubber - **ADJUST THIS BASED ON YOUR SCRUBBER VERSION'S DOCUMENTATION**
        command = [
            SCRUBBER_EXE_PATH,
            # Assuming Scrubber takes input and output file paths directly.
            # Common patterns:
            # scrubber.exe <inputfile> <outputfile>
            # scrubber.exe -i <inputfile> -o <outputfile>
            # scrubber.exe --input=<inputfile> --output=<outputfile>
            # The example uses -i and -o, ADJUST AS NEEDED!
            "-i", infile_path,
            "-o", outfile_path,
        ]
        command.extend(SCRUBBER_DEFAULT_OPTIONS)

        print(f"DEBUG: Running Scrubber command: {' '.join(command)}")
        # On Windows, Popen might need shell=True if the .exe path has spaces and isn't quoted
        # or if it's not directly executable without shell context, but try without first.
        # Using shell=True has security implications if command parts are from untrusted input.
        # Here, command parts are constructed by us, so it's safer.
        # However, it's better if the .exe can be called directly.

        # For Windows .exe, sometimes providing full path and ensuring it's executable is enough.
        # If SCRUBBER_EXE_PATH contains spaces, it MUST be handled carefully,
        # often by ensuring it's the first element and subprocess handles quoting, or by using shell=True
        # (though shell=True should be a last resort).

        # Using startupinfo to prevent console window from popping up on Windows
        startupinfo = None
        if os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # startupinfo.wShowWindow = subprocess.SW_HIDE # To hide the window

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        stdout, stderr = process.communicate(timeout=120)

        if process.returncode == 0:
            print("DEBUG: Scrubber executed successfully.")
            # Now read the output file
            if os.path.exists(outfile_path):
                with open(outfile_path, "r", encoding="utf-8") as f_read_final:
                    deidentified_text = f_read_final.read()
                if not deidentified_text.strip() and text_content.strip():
                    print("WARN: Scrubber output was empty, but input was not. Check Scrubber logs/config.")
                    return f"[SCRUBBER_WARN: EMPTY OUTPUT] {text_content}"
                return deidentified_text
            else:
                print(f"ERROR: Scrubber output file '{outfile_path}' not found after successful execution.")
                return f"[SCRUBBER_ERROR: OUTPUT FILE MISSING] {text_content}"
        else:
            print(f"ERROR: Scrubber execution failed. Return code: {process.returncode}")
            print(f"Scrubber STDOUT: {stdout.decode(sys.stdout.encoding or 'utf-8', errors='replace')}")
            print(f"Scrubber STDERR: {stderr.decode(sys.stderr.encoding or 'utf-8', errors='replace')}")
            return f"[SCRUBBER_ERROR: EXECUTION FAILED - CODE {process.returncode}] {text_content}"

    except FileNotFoundError:
        print(
            f"ERROR: Scrubber executable '{SCRUBBER_EXE_PATH}' not found or not executable. Ensure it's in PATH or the path is correct.")
        return f"[SCRUBBER_CONFIG_ERROR: EXE NOT FOUND/EXECUTABLE] {text_content}"
    except subprocess.TimeoutExpired:
        print(f"ERROR: Scrubber process timed out.")
        if hasattr(process, 'kill'): process.kill()
        # stdout, stderr = process.communicate() # This might hang if already timed out
        print(
            f"Scrubber STDOUT (on timeout): {stdout.decode(sys.stdout.encoding or 'utf-8', errors='replace') if stdout else 'N/A'}")
        print(
            f"Scrubber STDERR (on timeout): {stderr.decode(sys.stderr.encoding or 'utf-8', errors='replace') if stderr else 'N/A'}")
        return f"[SCRUBBER_TIMEOUT_ERROR] {text_content}"
    except Exception as e:
        print(f"ERROR during Scrubber de-identification: {e}")
        import traceback
        traceback.print_exc()
        return f"[SCRUBBER_EXCEPTION: {type(e).__name__}] {text_content}"
    finally:
        if 'infile_path' in locals() and os.path.exists(infile_path): os.remove(infile_path)
        if 'outfile_path' in locals() and os.path.exists(outfile_path): os.remove(outfile_path)


# Keep basic_deidentify_text as a fallback or alternative
def basic_deidentify_text(text):
    if not text: return ""
    text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', '[NAME_REDACTED]', text)
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr)\.\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', '[NAME_REDACTED]', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE_REDACTED]', text)
    text = re.sub(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{2,4})?\b',
        '[DATE_REDACTED]', text)
    text = re.sub(r'\b(age\s+\d{1,3}|\d{1,3}[-\s]year[s]?[-\s]old)\b', '[AGE_REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b', '[PHONE_REDACTED]', text)
    text = re.sub(
        r'\b\d{3,5}\s+[A-Z0-9][A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b',
        '[ADDRESS_REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)  # Basic SSN
    text = re.sub(r'\b[A-Z]{1,2}\d{5,}[A-Z]?\b', '[ID_REDACTED]', text)  # Generic ID like MRN
    print("DEBUG: Basic de-identification applied.")
    return text