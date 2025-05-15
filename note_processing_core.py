# note_processing_core.py
import sys
import io

# --- UTF-8 Setup ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# --- End UTF-8 Setup ---

from google.cloud import aiplatform  # Ensure this is imported
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
from Bio import Entrez
import os

PROJECT_ID = os.getenv("AI_PROJECT_ID", "amy-lloyd")
LOCATION = os.getenv("AI_LOCATION", "us-central1")

STABLE_MODEL_NAME = "gemini-1.0-pro"  # Or your confirmed working model like "gemini-pro"

YOUR_DETAILED_INSTRUCTIONS = """Use the entirety This GPT acts as a first-year neurology resident, trained to extract and synthesize neurological case information from clinical texts such as progress notes or transcripts. Before starting a case, the GPT prompts the user to provide the reason for the current visit and any relevant comments. Uploaded or pasted documents are assumed to be from prior visits and used only as historical context. The GPT then fills out a detailed, structured neurology case template. The HPI must be at least three paragraphs. All clinical sections must be completed, including:

- Timeline of Symptoms
- HPI
- Other Risk Factors
- Family History
- Associated Findings
- Sleep, Mood, Functional Considerations, Potential Harms
- Response to Past Treatment
- Previous Workup: MRI, FDG-PET, Neuropsychological Exam, Labs, Genetics
- Cognitive symptoms: Memory Function, Pattern, Severity, Attention, Executive Function, Visuospatial Function, Social Cognition, Language
- Motor symptoms: Gait, extrapyramidal/cerebellar/pyramidal symptoms, dysarthria, swallow, balance, autonomic function, olfaction, fits
- NPI-Q: Must be formatted as a table with symptom, severity (0–3), and distress (0–5), with a total row
- Functional assessment: IADLs and ADLs scored 0–3 and completed
- Supervision needs
- Medical capacity evaluation
- Driving safety risks: Assess based on provided information. If concerns like somnolence, poor attention, or visuospatial issues are noted in the input, state the risk explicitly. If a Mayo Scale score is provided in the input, include it. If no relevant information for driving risk is present in the input, write "NO INFORMATION FOUND".
- Past Medical/Surgical History, Allergies, Medications (including at-risk)
- Social History
- Review of Systems
- Physical Exam & Neurological Exam with relevant negatives
- Visit Vitals

**[YOUR SEPARATELY GENERATED DETAILED MEDICAL EXPLANATION, DIFFERENTIALS, AND PLAN WILL BE INSERTED HERE BY THE SCRIPT]**

- After the separately generated plan, generate a patient instruction section in plain language. This explainer should be written directly within this note, drawing relevant information from the separately generated "Medical Explanation" and "Plan" (which will be effectively part of the context once inserted by the script).
- Rewrite the content for the patient instructions in plain, non-technical English suitable for an average 8th-grade reader.
- Organize the patient instruction information into these clear sections:
    What We Found (Diagnosis/Assessment)
    What This Means
    What We’re Doing About It (Treatment Plan)
    What You Can Do (Patient Instructions)
    What to Watch Out For
    When to Call Us
Use a warm, supportive tone. Avoid medical jargon or explain it clearly when necessary.
Keep each section concise but informative. Aim for clarity and reassurance.
End with a gentle summary of next steps and encouragement.

If MMSE, FAQ, or CDR are present in the input, results should be formatted in visually clean and readable tables. If any section (excluding the placeholder for Medical Explanation, Plan, and related Patient Instructions) lacks data from the input, write “NO INFORMATION FOUND.” When content is ambiguous, ask clarifying questions. Section headings should be the same size as body text. Tables should use alternating row shading and bold headers. Emojis and icons must not be used. The note must begin with the header "UT Health San Antonio" styled to be visually prominent.
Do not include dates in the note.

**Alzheimer's Disease Candidate Checklist Instructions (Revised):**
Include the 'Alzheimer's Disease Candidate Checklist' section (using the template below verbatim)
IF your initial assessment of the provided 'PATIENT DATA' (especially Background Info, Transcription, and User Insights) suggests that
'Alzheimer's Disease' or 'Mild Cognitive Impairment due to Alzheimer's Disease' is a probable or possible consideration for this patient.
This checklist should appear immediately BEFORE the final signature section of this main note.
If Alzheimer's Disease is not considered probable or possible based on the *initial input data and your initial assessment*, OMIT this entire checklist section.

When including the checklist, Critically review the input text provided.
If specific data corresponding to a checklist item (e.g., MMSE score, GDS score, APOE status, BMI, specific exclusionary criteria answers)
is present in the input, YOU MUST replace the '***' placeholder or select the correct 'No / YES' option using that data.
If the specific data for an item is NOT in the input, leave the '***' placeholder or the 'No *** YES' options exactly as they appear in the template. Do not make up data.

**Alzheimer's Disease Candidate Checklist Template (This is the template the first LLM might include if AD is suspected. The script will then use a clean version for dynamic filling):**
--- Alzheimer's Disease Candidate Checklist ---
<INCLUSION CRITERIA>
1- Age between 50 - 90 years,
2- Mild cognitive impairment (MCI) OR Mild Dementia due to Alzheimer's Disease
3- Amyloid PET or CSF Biomarkers positive for beta Amyloid.
4- MMSE > 19.
</INCLUSION CRITERIA>

<OTHER DATA- include ONLY if INCLUSION CRITERIA SATISFIED>
- MMSE score >21: ***/30. List the subsection scores if available. IF NOT insert a template to assess MMSE.
- MoCA: *** /30 List the subsection scores if available. IF NOT insert a template to assess MOCA
- CDR-global: 0.5 *** 1.0 List the subsection scores if available to assess CDR
- FAQ: ***/30 List the subsection scores if available to assess FAQ.
Baseline laboratory date: CBC, CMP, TSH, B12, Folate,
Vitamin D, INR, aPTT. List results or list labs to be ordered.
APOE status: *** (e.g., e3/e4, e4/e4, or "Information not found" or "To be ordered")
If APOE status known, state associated ARIA risk: ***
</OTHER DATA>

<EXCLUSION CRITERIA- include ONLY if INCLUSION CRITERIA SATISFIED>
Does the patient have any of the following (A
"YES" is exclusionary):

No *** YES Physical, Mental, or Neurological Issue
Contribution to Cognitive Impairment
No *** YES Age Criteria: Individuals younger than
50 years old or older than 90 years.
No *** YES MRI Contraindication: Any medical or
physiological condition that would make undergoing Magnetic Resonance Imaging
(MRI) inadvisable or dangerous for the patient. Some of these contraindications
are older pacemakers, neurostimulators, cochlear implants, insulin pumps,
intrathecal drug pumps, metallic foreign bodies, older aneurysm clips, older
tattoos, and claustrophobia.
No *** YES MRI
Evidence of Neurological Abnormalities: MRI results displaying more than four
microhemorrhages, superficial siderosis, the presence of vasogenic edema,
indications of recent or ongoing strokes, aneurysms, vascular malformations,
extensive white matter disease, neoplasms, evidence of infections, or any other
neurological anomaly that, in the opinion of the physician, significantly
heighten the risk of medical complications.
No *** YES Bleeding
Risks and Medication: Exclusionary conditions include untreated disorders that
lead to excessive bleeding, a platelet count that is below 50 or an
International Normalized Ratio (INR) above 1.5. Anticoagulation is also
exclusionary.
No *** YES Oncological
Status: The presence of an active cancerous condition or malignancy, except
after prolonged remission at the discretion of the prescribing physician.
No *** YES Mental
Health Conditions: The presence of untreated or uncontrolled depression or
anxiety disorders, any other unstable psychiatric conditions, any current
history of suicidal thoughts or actions in the preceding five years, or a
Geriatric Depression Scale (GDS) score exceeding 8 will lead to exclusion. GDS:
***/15
No *** YES Substance Dependency: Any ongoing misuse or
dependence on drugs or alcohol.
N/A *** YES Reproductive
Status: Women who are currently pregnant or are in the lactation phase
post-childbirth.
No *** YES Body Mass Index (BMI) Range: Individuals
with a BMI that exceeds 35 or is less than 17. Estimated body mass index is
*** kg/m² as calculated from the following: ***
No *** YES Immunological Disease Diagnosis:
Individuals diagnosed with a systemic immunological disorder, irrespective of
the treatment they are on.
No *** YES HIV Status: The presence of Human
Immunodeficiency Virus (HIV) infection – if unknown and the patient is in a
high risk group, please order an HIV test [needs a consent]
No *** YES Recent Neurological Episodes: Any history
of strokes, seizures, or transient ischemic attacks occurring within the last
year. Or seizure with late life onset.
No *** YES Current Use of Specific Medications:
Ongoing usage of systemic monoclonal antibodies, immunoglobulin, and other
biological treatments.
</EXCLUSION CRITERIA>
--- (End Checklist Template) ---

Conclude every note with this signature.
"""

ALZHEIMERS_CHECKLIST_TEMPLATE_FOR_PROCESSING = """--- Alzheimer's Disease Candidate Checklist ---
<INCLUSION CRITERIA>
1- Age between 50 - 90 years: ***
2- Cognitive Status (MCI or Mild Dementia due to AD): ***
3- Biomarker Evidence (Amyloid PET or CSF positive for beta Amyloid): ***
4- MMSE Score > 19: ***
Overall Inclusion Criteria Met: *** (YES/NO/UNCERTAIN based on items 1-4)
</INCLUSION CRITERIA>

<OTHER DATA- include ONLY if INCLUSION CRITERIA SATISFIED or UNCERTAIN and AD is primary Dx>
- MMSE score: ***/30. (Subsection scores: ***)
- MoCA score: ***/30. (Subsection scores: ***)
- CDR-global: *** (Score: *** ; Subsection scores: ***)
- FAQ score: ***/30. (Subsection scores: ***)
- Baseline Labs (CBC, CMP, TSH, B12, Folate, Vit D, INR, aPTT): Results: *** (or "To be ordered" or "No significant abnormalities noted")
- APOE Genotype: *** (e.g., e3/e3, e3/e4, e4/e4, or "Not available" or "To be ordered")
- Associated ARIA Risk (if APOE known and applicable): ***
</OTHER DATA>

<EXCLUSION CRITERIA- include ONLY if INCLUSION CRITERIA SATISFIED or UNCERTAIN and AD is primary Dx>
For each item, select "No" (criterion not met, patient NOT excluded by this), "YES" (criterion met, patient IS excluded by this), or "UNKNOWN" (insufficient information).

No / YES / UNKNOWN : Physical, Mental, or Neurological Issue (other than primary AD) significantly contributing to Cognitive Impairment.
No / YES / UNKNOWN : Age Criteria (younger than 50 or older than 90 years). (Actual Age: ***)
No / YES / UNKNOWN : MRI Contraindication.
No / YES / UNKNOWN : MRI Evidence of significant exclusionary Neurological Abnormalities (>4 microhemorrhages, superficial siderosis, vasogenic edema, recent/ongoing strokes, aneurysms, vascular malformations, extensive white matter disease, neoplasms, infections, etc.).
No / YES / UNKNOWN : Bleeding Risks (untreated bleeding disorder, platelets < 50, INR > 1.5, or current anticoagulation).
No / YES / UNKNOWN : Oncological Status (active cancer, unless prolonged remission at physician discretion).
No / YES / UNKNOWN : Mental Health Conditions (untreated/uncontrolled depression/anxiety, other unstable psychiatric conditions, suicidality in past 5 years, GDS > 8). (GDS Score: ***/15)
No / YES / UNKNOWN : Substance Dependency (ongoing drug or alcohol misuse/dependence).
No / YES / UNKNOWN : Reproductive Status (women currently pregnant or lactating). (If N/A, state N/A)
No / YES / UNKNOWN : Body Mass Index (BMI) outside range 17-35. (Estimated BMI: *** kg/m² ; Calculated from: ***)
No / YES / UNKNOWN : Immunological Disease Diagnosis (diagnosed systemic immunological disorder).
No / YES / UNKNOWN : HIV Status (positive HIV; if unknown and high risk, needs test).
No / YES / UNKNOWN : Recent Neurological Episodes (stroke, seizure, TIA within last year; or late-life onset seizure).
No / YES / UNKNOWN : Current Use of Specific Medications (systemic monoclonal antibodies, immunoglobulin, other biological treatments).

Overall Exclusion Criteria Summary: *** (e.g., "No exclusionary criteria definitively met." / "Exclusionary criteria X, Y met." / "Insufficient information to rule out all exclusions.")
</EXCLUSION CRITERIA>
--- (End Checklist Template) ---"""

SIGNATURE = """
Arash Salardini, MD
Klesse Foundation Distinguished Chair in Alzheimer and Neurodegenerative Diseases
Associate Professor, Cognitive and Behavioral Neurology
Chief, Division of Cognitive and Behavioral Neurology
Director, Behavioral Neurology and Neuropsychiatry Fellowship Program
Glenn Biggs Institute for Alzheimer's & Neurodegenerative Diseases
UT Health San Antonio
8300 Floyd Curl Drive
San Antonio, TX 78229
Office: 210-450-9700
Fax: 210-450-6039
"""


# --- All your existing functions from the previous note_processing_core.py ---
# initialize_vertex_ai, fetch_recent_guidelines, summarize_literature_with_gemini,
# generate_neurology_note_body (will be updated below),
# generate_diagnostic_assessment_llm (will be updated below),
# extract_diagnosis_with_llm, is_alzheimers_primary_diagnosis,
# process_alzheimers_checklist_with_llm,
# generate_patient_specific_criteria_elaboration,
# generate_missing_info_summary
# --- Ensure they are all here ---

def initialize_vertex_ai():
    # ... (same as before)
    print("DEBUG: Initializing Vertex AI...")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"Vertex AI initialized for project '{PROJECT_ID}' in location '{LOCATION}'.")
    except Exception as e:
        print(f"CRITICAL ERROR initializing Vertex AI: {e}")
        raise


def fetch_recent_guidelines(diagnosis, email="salardini@uthscsa.edu", max_results=5):
    # ... (same as before)
    print(f"DEBUG: Searching PubMed for guidelines related to: {diagnosis}")
    Entrez.email = email
    search_term = f'("{diagnosis}"[MeSH Terms] OR "{diagnosis}"[Title/Abstract]) AND ("guideline"[Publication Type] OR "practice guideline"[Publication Type])'
    try:
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=str(max_results), sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()
        ids = search_results["IdList"]
        count = int(search_results["Count"])
        print(f"DEBUG: Found {count} potential guidelines, fetching details for up to {len(ids)}.")
        if not ids:
            print("DEBUG: No relevant guideline IDs found on PubMed.")
            return []
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts_text = handle.read()
        handle.close()
        raw_abstracts = abstracts_text.strip().split("\n\n")
        processed_abstracts = []
        for i, abst in enumerate(raw_abstracts):
            clean_abst = re.sub(r"^\s*\d+\.\s*", "", abst)
            clean_abst = re.sub(r"^\s*PMID-\s*\d+\s*", "", clean_abst)
            if len(clean_abst) > 50:
                processed_abstracts.append(clean_abst.strip())
            if i < 10:  # Log only a few for brevity
                print(f"DEBUG: Raw abstract part {i}: {abst[:100]}...")
        print(f"DEBUG: Fetched and processed {len(processed_abstracts)} abstracts from {len(raw_abstracts)} raw parts.")
        return processed_abstracts
    except Exception as e:
        print(f"ERROR during PubMed search/fetch: {e}")
        return []


def summarize_literature_with_gemini(abstracts, diagnosis):
    # ... (same as before)
    print("DEBUG: Summarizing literature with Gemini...")
    if not abstracts:
        return "### Recent Literature Summary\n\nNo recent relevant literature found or summarized."
    context_abstracts = []
    current_length = 0
    max_context_length = 15000
    for abst in abstracts:
        if len(context_abstracts) < 5 and (current_length + len(abst)) < max_context_length:
            context_abstracts.append(abst)
            current_length += len(abst)
        else:
            break
    if not context_abstracts:
        return "### Recent Literature Summary\n\nAbstracts were too long or none suitable for summary."
    context = "\n\n---\n\n".join(context_abstracts)
    literature_summary_instructions = f"""
    Based on the following abstracts related to '{diagnosis}', please provide a summary for a "Recent Literature Summary" section of a clinical note.
    Instructions:
    1. Review the provided abstracts, focusing on clinical guidelines or practice guidelines if present.
    2. Select 2-3 of the most relevant and recent guidelines mentioned in the abstracts.
    3. For each selected guideline, provide a concise summary of its key recommendations pertinent to '{diagnosis}'.
    4. Briefly comment on the relevance of each guideline to the diagnosis, prognosis, or management of a typical case involving '{diagnosis}'.
    5. Format the output as a numbered list, ensuring the section starts with the heading "### Recent Literature Summary".
    6. If fewer than 2 relevant guidelines are found in the provided abstracts, state that under the heading.
    Abstracts Provided:
    ---
    {context}
    ---
    Generate the "Recent Literature Summary" section now:
    """
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        response = model.generate_content([literature_summary_instructions])
        print("DEBUG: Received literature summary from Gemini.")
        generated_summary = response.text.strip()
        if not generated_summary.lower().startswith("### recent literature summary"):
            generated_summary = "### Recent Literature Summary\n\n" + generated_summary
        return generated_summary
    except Exception as e:
        print(f"ERROR during Gemini literature summarization: {e}")
        return "### Recent Literature Summary\n\nError summarizing literature."


def generate_neurology_note_body(gnb_historical_text, gnb_current_visit_context, gnb_user_insights):
    """
    Generates the main body of the neurology note.
    gnb_historical_text: Corresponds to "Background Information".
    gnb_current_visit_context: Corresponds to "Transcription" and other current visit elements.
    gnb_user_insights: Corresponds to "Additional Info" and "Revised Info".
    """
    print("DEBUG: Inside generate_neurology_note_body function.")
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        # Updated prompt to reflect new input structure
        prompt_content = f"""
        {YOUR_DETAILED_INSTRUCTIONS} 

        PATIENT DATA START
        ---
        I. BACKGROUND INFORMATION (Prior to this visit):
        {gnb_historical_text if gnb_historical_text else "Not provided."}
        ---
        II. CURRENT VISIT CONTEXT (Includes transcription if available, and reason for visit):
        {gnb_current_visit_context if gnb_current_visit_context else "Not provided."}
        ---
        III. USER INSIGHTS (Clinician's additional thoughts, emphasis, potential diagnoses, revisions):
        {gnb_user_insights if gnb_user_insights else "Not provided."}
        ---
        PATIENT DATA END

        Generate the neurology note structure now based on ALL the provided patient data and the general instructions.
        Ensure the HPI draws from the "CURRENT VISIT CONTEXT" including any transcription.
        Use "BACKGROUND INFORMATION" for historical context.
        Consider "USER INSIGHTS" for nuances, emphasis, or potential diagnoses to explore.
        Remember to include the placeholder "[YOUR SEPARATELY GENERATED DETAILED MEDICAL EXPLANATION AND PLAN WILL BE INSERTED HERE BY THE SCRIPT]"
        Do not generate the detailed medical explanation, differentials, or plan itself in this step.
        The Patient Instructions section should also be generated based on the understanding that a detailed Medical Explanation and Plan will be inserted later by the script.
        Do not include dates.
        """
        print("\n--- Sending main note body prompt to Gemini ---")
        response = model.generate_content([prompt_content])
        print("--- Received main note body response from Gemini ---")
        return response.text
    except Exception as e:
        print(f"An error occurred during main note body Gemini API call: {e}")
        return f"Error generating main note body: {e}"


def generate_diagnostic_assessment_llm(diag_historical_context, diag_reason_for_visit,
                                       diag_user_insights_and_revisions):
    """
    Generates the detailed diagnostic assessment.
    diag_historical_context: Background Info + Transcription.
    diag_reason_for_visit: High-level reason (can be generic as context is rich).
    diag_user_insights_and_revisions: Additional Info + Revised Info.
    """
    print("DEBUG: Attempting to generate diagnostic assessment (Medical Explanation, Plan) using LLM...")
    diagnostic_prompt_instructions = """You are an AI Neurological Diagnostic Assistant. Based on the patient history, neurological examination findings, and any available ancillary data that have been provided to you, please provide a comprehensive neurological assessment focused on cognitive disorders.
Requested Output:
Please structure your response as follows (This entire response will form the "Medical Explanation", "Plan Rationale", and "Plan" sections of a larger note, so ensure appropriate subheadings like "## Medical Explanation", "## Plan Rationale", "## Plan"):

## Medical Explanation

**Summary of Key Patient Findings:** Briefly summarize the most pertinent positive and negative findings from the patient's history, examination, and ancillary data that were provided.

**Most Likely Diagnosis:** YOU MUST STATE THE DIAGNOSIS IN THE FOLLOWING 3-PART FORMAT, EVEN IF SOME PARTS ARE UNKNOWN:
1- [Severity of dementia/impairment, e.g., Mild Dementia, Mild Cognitive Impairment, Moderate Dementia, Unknown Severity], 2- [Clinical Syndrome, e.g., Amnestic Presentation, Non-fluent PPA, Behavioral Variant FTD, Alzheimer's Clinical Syndrome, Unknown Syndrome], 3- [Suspected Underlying Pathology, e.g., Alzheimer's Disease, FTLD-tau, Vascular Disease, Unknown Pathology]. Example: "1- Mild Dementia, 2- Amnestic and Language Presentation, 3- Alzheimer's Disease." If a part is truly unknown from context, state "Unknown [PartName]".

    Pathophysiology: Describe the underlying biological mechanisms and structural/functional changes associated with this disorder.
    Natural History: Outline the typical untreated course of the disorder from onset over time.
    Semiology (Clinical Manifestations): Detail the common signs, symptoms, and clinical features characteristic of this disorder across its stages.
    Treatment (General Approaches): Describe the current standard therapeutic interventions, including pharmacological and non-pharmacological strategies. Do not prescribe, but describe general approaches.
    Progression: Explain the typical pattern and rate of decline or change associated with this disorder.
**Differential Diagnoses:** List other possible diagnoses in order of likelihood. For each differential diagnosis, include:
    Pathophysiology, Natural History, Semiology, Treatment (General Approaches), Progression (as above).
    Pros (Specific to this Patient): Specific findings from the provided patient information that support this diagnosis for this particular patient.
    Cons (Specific to this Patient): Specific findings from the provided patient information that argue against this diagnosis for this particular patient, or features that are atypical for this diagnosis based on the patient's presentation.
**Reasoning for Most Likely Diagnosis (Specific to this Patient):** Provide a detailed explanation.
**Missing Information and Recommended Next Steps (Specific to this Patient):** Identify crucial pieces of information.
**Prognostic Considerations (Specific to this Patient, if applicable based on likely diagnoses):** Briefly mention.

**## Plan Rationale**
Conclude this entire section with a paragraph explaining the rationale for the proposed clinical plan based on the assessment.

**## Plan**
Following the rationale, list the proposed clinical plan as a numbered list.

Important Caveats: Include a disclaimer that this is an AI-generated assessment based on the provided information and is not a substitute for evaluation and diagnosis by a qualified human neurologist. Emphasize that clinical correlation and further investigation are essential.
"""
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        # Updated prompt to reflect new input structure
        prompt = f"""{diagnostic_prompt_instructions}

        # Base your assessment on ALL the following provided patient data:

        PATIENT DATA START
        ---
        I. BACKGROUND AND CURRENT VISIT INFORMATION (Includes prior history and current visit transcription):
        {diag_historical_context if diag_historical_context else "Not provided."} 
        ---
        II. HIGH-LEVEL REASON FOR VISIT (For context, full details are above):
        {diag_reason_for_visit if diag_reason_for_visit else "Not provided."}
        ---
        III. USER INSIGHTS (Clinician's additional thoughts, emphasis, potential diagnoses, revisions):
        {diag_user_insights_and_revisions if diag_user_insights_and_revisions else "Not provided."}
        ---
        PATIENT DATA END

        Please generate the comprehensive neurological assessment now.
        Ensure the "Most Likely Diagnosis" STRICTLY follows the "1- Severity, 2- Syndrome, 3- Pathology" format.
        """
        print("DEBUG: Sending prompt to LLM for detailed diagnostic assessment...")
        response = model.generate_content([prompt])
        generated_assessment_text = response.text.strip()
        print(f"DEBUG: Received diagnostic assessment (length: {len(generated_assessment_text)})")
        if not generated_assessment_text.strip().lower().startswith("## medical explanation"):
            generated_assessment_text = "## Medical Explanation\n\n" + generated_assessment_text
        return generated_assessment_text
    except Exception as e:
        print(f"ERROR during LLM diagnostic assessment generation: {e}")
        return "## Medical Explanation\n\n[Error generating diagnostic assessment.]\n\n## Plan Rationale\n\n[Error generating plan rationale.]\n\n## Plan\n\n[Error generating plan.]"


def extract_diagnosis_with_llm(diagnostic_assessment_text):
    # ... (same as before, relies on the output of generate_diagnostic_assessment_llm)
    print("DEBUG: Attempting to extract diagnosis from diagnostic_assessment_text...")
    mld_regex = re.compile(
        r"^\s*\*\*(?:Most\s+Likely\s+Diagnosis|MOST\s+LIKELY\s+DIAGNOSIS):\s*\*\*"
        r"\s*1-\s*(?P<severity>[^,]+(?:,\s*[^,]+)*?)\s*,"
        r"\s*2-\s*(?P<syndrome>[^,]+(?:,\s*[^,]+)*?)\s*,"
        r"\s*3-\s*(?P<pathology>[^,.]+?)\s*\.?\s*$",
        re.MULTILINE | re.IGNORECASE
    )
    for line in diagnostic_assessment_text.splitlines():
        match = mld_regex.search(line.strip())
        if match:
            severity = match.group("severity").strip()
            syndrome = match.group("syndrome").strip()
            pathology = match.group("pathology").strip()
            print(f"DEBUG: Regex MLD parts: Severity='{severity}', Syndrome='{syndrome}', Pathology='{pathology}'")
            if "unknown pathology" not in pathology.lower() and pathology and "unknown" not in pathology.lower():
                if "alzheimer's disease" in pathology.lower() or "ad" in pathology.lower():
                    if "mci" in severity.lower() or "mild cognitive impairment" in severity.lower():
                        return "Mild Cognitive Impairment due to Alzheimer's Disease"
                    return "Alzheimer's Disease"
                return pathology
            elif "alzheimer's clinical syndrome" in syndrome.lower() and "unknown pathology" in pathology.lower():
                return "Alzheimer's Disease"
            elif syndrome and "unknown syndrome" not in syndrome.lower():
                return f"{severity} - {syndrome}" if severity and "unknown severity" not in severity.lower() else syndrome
            elif severity and "unknown severity" not in severity.lower():
                return severity
            else:
                print(
                    "DEBUG: Regex matched MLD line structure but parts were empty, unsuitable, or only contained terms like 'unknown'.")
                break

    print("DEBUG: Regex for structured MLD failed or did not yield a clear diagnosis. Falling back to LLM extraction.")
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        prompt = f"""From the following clinical assessment text, find the section explicitly titled "**Most Likely Diagnosis:**" or "Most Likely Diagnosis:".
        Extract the single most specific primary diagnosis stated immediately after this title.
        Prioritize a diagnosis that includes "Alzheimer's Disease" or "Mild Cognitive Impairment due to Alzheimer's Disease" if present and clearly stated as the most likely.
        If a 3-part diagnosis (Severity, Syndrome, Pathology) is given, extract the most clinically relevant part, usually the Pathology (e.g., "Alzheimer's Disease" from "..., 3- Alzheimer's Disease"). If pathology is "Unknown" but syndrome points to AD, use that.
        Respond with ONLY the diagnosis name (e.g., "Alzheimer's Disease", "Mild Cognitive Impairment due to Alzheimer's Disease", "Frontotemporal Dementia").
        Do not include "1-", "2-", "3-", surrounding text, explanations, or the heading "Most Likely Diagnosis:" itself.
        If this specific section or a clear single diagnosis after the title is not found, or if it says "Leave blank if uncertain" and is blank, respond with "NONE".

        Clinical Assessment Text to Analyze:
        ---
        {diagnostic_assessment_text}
        ---

        Primary diagnosis:"""
        print("DEBUG: Sending prompt to LLM for diagnosis extraction from assessment...")
        response = model.generate_content([prompt])
        extracted_text = response.text.strip()
        print(f"DEBUG: LLM raw response for diagnosis: '{extracted_text}'")
        if extracted_text.endswith('.'):
            extracted_text = extracted_text[:-1].strip()
        if not extracted_text or extracted_text.upper() == "NONE" or len(extracted_text) > 150:
            print(f"DEBUG: LLM indicated no clear primary diagnosis found or response unsuitable: '{extracted_text}'")
            return None
        else:
            extracted_text = re.sub(r"^\s*\d+-\s*", "", extracted_text).strip()
            if "likely due to ad" in extracted_text.lower() or \
                    "possible ad" in extracted_text.lower() or \
                    "alzheimer's disease" in extracted_text.lower() or \
                    "alzheimer's clinical syndrome" in extracted_text.lower():
                if "mci" in extracted_text.lower() or "mild cognitive impairment" in extracted_text.lower():
                    extracted_text = "Mild Cognitive Impairment due to Alzheimer's Disease"
                else:
                    extracted_text = "Alzheimer's Disease"
            print(f"DEBUG: Extracted diagnosis (LLM fallback): {extracted_text}")
            return extracted_text
    except Exception as e:
        print(f"ERROR during LLM diagnosis extraction from assessment: {e}")
        return None


def is_alzheimers_primary_diagnosis(diagnosis_text):
    # ... (same as before)
    if not diagnosis_text:
        return False
    ad_keywords = ["alzheimer's disease", "ad", "mild cognitive impairment due to alzheimer's"]
    diagnosis_lower = diagnosis_text.lower()
    for keyword in ad_keywords:
        if keyword in diagnosis_lower:
            if "not alzheimer" in diagnosis_lower or "rule out alzheimer" in diagnosis_lower or "unlikely alzheimer" in diagnosis_lower:
                continue
            return True
    return False


def process_alzheimers_checklist_with_llm(checklist_template_to_fill, background_info, transcription_context,
                                          user_insights, diagnostic_assessment_text):
    # ... (Prompt inside uses these distinct parameters)
    print("DEBUG: Processing Alzheimer's Checklist with LLM (New Exclusion Format)...")
    full_context = f"""
    PATIENT DATA:
    I. BACKGROUND INFORMATION (Prior to this visit):
    {background_info if background_info else "Not provided."}
    ---
    II. CURRENT VISIT TRANSCRIPTION (and other current visit context):
    {transcription_context if transcription_context else "Not provided."}
    ---
    III. USER INSIGHTS (Clinician's additional thoughts, emphasis, potential diagnoses, revisions):
    {user_insights if user_insights else "Not provided."}
    ---
    IV. FULL DIAGNOSTIC ASSESSMENT (includes 'Most Likely Diagnosis', 'Plan', etc.):
    {diagnostic_assessment_text}
    """
    # The prompt for this function (as provided in the previous full script)
    # should already be structured to take this `full_context`.
    # I'm re-pasting the prompt here for completeness of this function.
    prompt = f"""
    You are tasked with meticulously populating an Alzheimer's Disease Candidate Checklist based on ALL provided patient information and the comprehensive diagnostic assessment.
    The checklist template is provided below.

    **Instructions for populating the checklist:**

    1.  **Review ALL provided patient data and the full diagnostic assessment (provided in CONTEXT below).**
    2.  **Analyze INCLUSION CRITERIA first:**
        *   For each item (1-4) in the `<INCLUSION CRITERIA>` section, determine if the patient meets the criterion based *solely* on the information provided IN THE CONTEXT.
        *   Replace `***` with specific data if found (e.g., "YES (Age: 72)" or "NO (MMSE: 15)" or "UNKNOWN"). For item 3 (Biomarker), if PET/CSF data is mentioned, state result (e.g., "YES (Amyloid PET positive for beta Amyloid)" or "NO (CSF normal for beta Amyloid)" or "UNKNOWN (No biomarker data for beta Amyloid)").
        *   For "Overall Inclusion Criteria Met:", explicitly state "YES" if all items 1-4 are definitively met. State "NO" if any item 1-4 is definitively not met. State "UNCERTAIN" if one or more items are unknown but others are met (especially if AD is the primary diagnosis).

    3.  **If "Overall Inclusion Criteria Met:" is "YES" or "UNCERTAIN" (and AD is the primary suspected diagnosis from the assessment):**
        *   Proceed to populate the `<OTHER DATA>` section. Replace `***` with specific values if found from CONTEXT. For subsection scores or lab results, list them if available or state "Not available" or "Not specified". For APOE, state genotype and then the ARIA risk on the next line if applicable based on the genotype shown in the template. If APOE is not available, state "Not available" or "To be ordered".
        *   Proceed to populate the `<EXCLUSION CRITERIA>` section. For each "No / YES / UNKNOWN" item:
            *   Carefully evaluate the patient data IN THE CONTEXT against the criterion.
            *   Replace "No / YES / UNKNOWN" with **ONLY ONE** of these three options: "No", "YES", or "UNKNOWN".
            *   Fill in any other `***` placeholders (like Actual Age, GDS Score, BMI) with data if available from CONTEXT, otherwise leave them as `***` or state "Not available".
        *   For "Overall Exclusion Criteria Summary:", provide a concise statement like: "No exclusionary criteria definitively met based on available information.", or "Exclusionary criterion X met.", or "Insufficient information to rule out all potential exclusions (see UNKNOWN items above)."

    4.  **If "Overall Inclusion Criteria Met:" is "NO":**
        *   Populate the `<INCLUSION CRITERIA>` section, clearly indicating why they are not met (e.g., "NO (Age: 45)").
        *   For the `<OTHER DATA>` and `<EXCLUSION CRITERIA>` sections, replace their entire content with a single line: "NOT APPLICABLE - Inclusion criteria not met."

    5.  **General Formatting:**
        *   Retain the exact structure and headings of the template.
        *   Be precise. Do not infer information. Use only the provided text from CONTEXT.

    **CONTEXT (Patient Data and Full Diagnostic Assessment):**
    ---
    {full_context}
    ---

    **ALZHEIMER'S DISEASE CANDIDATE CHECKLIST TEMPLATE TO POPULATE:**
    ---
    {checklist_template_to_fill}
    ---

    Populate the checklist now:
    """
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        response = model.generate_content([prompt])
        processed_checklist = response.text.strip()
        print("DEBUG: Received processed Alzheimer's checklist from LLM (New Exclusion Format).")

        checklist_start_index = processed_checklist.find("--- Alzheimer's Disease Candidate Checklist ---")
        if checklist_start_index != -1:
            processed_checklist = processed_checklist[checklist_start_index:]
        else:
            processed_checklist = "--- Alzheimer's Disease Candidate Checklist ---\n" + processed_checklist + "\n--- (End Checklist Template) ---"
        return processed_checklist
    except Exception as e:
        print(f"ERROR during LLM checklist processing: {e}")
        return checklist_template_to_fill


def generate_patient_specific_criteria_elaboration(primary_diagnosis, background_info, transcription_context,
                                                   user_insights, diagnostic_assessment_text):
    # ... (Prompt inside uses these distinct parameters)
    if not is_alzheimers_primary_diagnosis(primary_diagnosis):
        return ""

    print(f"DEBUG: Generating patient-specific criteria elaboration for Dx: {primary_diagnosis}...")
    full_context_for_elaboration = f"""
    PATIENT DATA:
    I. BACKGROUND INFORMATION (Prior to this visit):
    {background_info if background_info else "Not provided."}
    ---
    II. CURRENT VISIT TRANSCRIPTION (and other current visit context):
    {transcription_context if transcription_context else "Not provided."}
    ---
    III. USER INSIGHTS (Clinician's additional thoughts, emphasis, potential diagnoses, revisions):
    {user_insights if user_insights else "Not provided."}
    ---
    IV. FULL DIAGNOSTIC ASSESSMENT (includes 'Most Likely Diagnosis', 'Plan', etc.):
    {diagnostic_assessment_text}
    """
    # The prompt for this function (as provided in the previous full script)
    # should already be structured to take this `full_context_for_elaboration`.
    # I'm re-pasting the prompt here for completeness of this function.
    prompt = f"""
    The patient's primary diagnosis appears to be '{primary_diagnosis}'.
    Based on ALL the provided patient data and the full diagnostic assessment (see CONTEXT below), provide a section titled "### Patient-Specific Elaboration of Diagnostic Criteria for Alzheimer's Disease".

    Instructions:
    1.  List the key diagnostic criteria for Alzheimer's Disease (e.g., from NIA-AA or DSM-5 for Major/Mild Neurocognitive Disorder due to AD). You can choose a standard set of criteria (e.g., NIA-AA for probable AD).
    2.  For EACH key criterion, briefly explain the criterion itself in one sentence.
    3.  Then, for EACH criterion, analyze the provided patient information IN THE CONTEXT (history, cognitive scores, imaging, biomarkers if mentioned, symptoms) and explicitly state how the patient's specific findings either SUPPORT or DO NOT SUPPORT (or if information is LACKING for) that particular criterion. Be specific and quote or refer to data points from the context.
    4.  Use bullet points for each main criterion. Under each main criterion, use sub-bullets for its explanation and then for the patient-specific analysis.
    5.  Maintain a clinical and objective tone.

    Example for one criterion:
    *   **Insidious onset and gradual progression of cognitive impairment:**
        *   *Explanation:* This criterion means the cognitive symptoms started slowly and have worsened steadily over time, rather than appearing abruptly or having a fluctuating course.
        *   *Patient Specifics:* The patient's family reports a decline "over the past two years" with "worsening memory and word-finding" (SUPPORTS gradual progression). The provided history does not mention any sudden neurological events that would suggest an acute onset (SUPPORTS insidious onset).

    CONTEXT (Patient Data and Full Diagnostic Assessment):
    ---
    {full_context_for_elaboration}
    ---

    Generate the "### Patient-Specific Elaboration of Diagnostic Criteria for Alzheimer's Disease" section now:
    """
    try:
        model = GenerativeModel(STABLE_MODEL_NAME)
        response = model.generate_content([prompt])
        elaboration_text = response.text.strip()
        print(f"DEBUG: Received 'Patient-Specific Criteria Elaboration' from Gemini.")
        if not elaboration_text.lower().startswith("### patient-specific elaboration"):
            elaboration_text = "### Patient-Specific Elaboration of Diagnostic Criteria for Alzheimer's Disease\n" + elaboration_text
        return elaboration_text
    except Exception as e:
        print(f"ERROR during patient-specific criteria elaboration generation: {e}")
        return "### Patient-Specific Elaboration of Diagnostic Criteria for Alzheimer's Disease\n\n[Error generating this section.]"


def generate_missing_info_summary(note_text_to_analyze):
    # ... (same as before)
    print("DEBUG: Generating Summary of Missing Information from combined note...")
    missing_sections = []
    pattern = r"^\s*(?:[#\*]+\s*)?(.*?)(?:\s*:\s*)?[\r\n]+\s*NO INFORMATION FOUND"
    matches = re.findall(pattern, note_text_to_analyze, re.MULTILINE | re.IGNORECASE)
    if matches:
        for header_line_candidate in matches:
            clean_header = header_line_candidate.strip()
            clean_header = re.sub(r"[:\*]+$", "", clean_header).strip()
            clean_header = re.sub(r"^[#\*]+\s*", "", clean_header).strip()
            if clean_header:
                exclusions = [
                    "medical explanation", "plan rationale", "plan", "patient instructions",
                    "recent literature summary", "plain language summary",
                    "summary of key patient findings", "differential diagnoses",
                    "reasoning for most likely diagnosis", "missing information and recommended next steps",
                    "prognostic considerations", "clinical plan", "important caveats",
                    "alzheimer's disease candidate checklist template",
                    "alzheimer's disease candidate checklist instructions",
                    "alzheimer's disease candidate checklist",
                    "supporting criteria from diagnostic algorithm",
                    "patient-specific elaboration of diagnostic criteria"
                ]
                is_excluded = False
                temp_clean_header_lower = clean_header.lower()
                for ex in exclusions:
                    if temp_clean_header_lower == ex:
                        is_excluded = True
                        break
                if not is_excluded:
                    if '\n' not in clean_header and len(clean_header) < 100:
                        missing_sections.append(clean_header)
                    else:
                        print(
                            f"DEBUG: Skipped potential multi-line/long header capture for missing info: '{clean_header[:100]}...'")

    summary_text = "## Summary of Missing Information\n\n"
    if missing_sections:
        unique_missing_sections = sorted(list(set(missing_sections)))
        summary_text += "The following sections were marked 'NO INFORMATION FOUND' in the initial note generation or specific data points were not available in the provided context:\n"
        for section in unique_missing_sections:
            summary_text += f"* {section}\n"
        print(f"DEBUG: Found missing sections for summary: {unique_missing_sections}")
    else:
        summary_text += "All relevant sections in the main note appeared to contain information or did not match 'NO INFORMATION FOUND' criteria for this summary.\n"
        print(
            "DEBUG: No specific 'NO INFORMATION FOUND' markers matched for summary, or all matched sections were excluded.")

    summary_text += "\nFor the Alzheimer's Checklist (if included), please review it directly for any '***' placeholders or items marked 'UNKNOWN' indicating data not found in the provided context for those specific checklist items."
    return summary_text.strip()


def generate_full_note(background_info_text, additional_info_text, transcription_text, revised_info_text):
    """
    Main function to generate the complete neurology note.
    Takes text from the four input categories.
    """
    print("DEBUG: Core note processing started.")
    # Vertex AI should ideally be initialized once when the Flask app starts.
    # If this function is called multiple times per app run, initializing here is okay too.
    # For a CLI tool, initializing here is standard.
    try:
        initialize_vertex_ai()
    except Exception as e_init:
        print(f"ERROR: Vertex AI Initialization Failed in generate_full_note: {e_init}")
        return f"ERROR: Vertex AI Initialization Failed: {e_init}"

    # --- Prepare context for LLMs ---
    gnb_historical_text = f"BACKGROUND INFORMATION (Prior to this visit):\n{background_info_text if background_info_text else 'Not provided.'}"
    gnb_current_visit_context = f"CURRENT VISIT CONTEXT (Includes transcription if available, and reason for visit):\n{transcription_text if transcription_text else 'Not provided.'}\n"
    if not transcription_text and not (
            "reason for visit" in gnb_current_visit_context.lower()):  # Add a placeholder if no transcription
        gnb_current_visit_context += "Primary reason for visit to be inferred from other context or user insights."

    gnb_user_insights = (
        f"USER'S ADDITIONAL INFORMATION/EMPHASIS (Clinician's thoughts, potential diagnoses, etc.):\n"
        f"{additional_info_text if additional_info_text else 'Not provided.'}\n\n"
        f"USER'S REVISED INFORMATION/FURTHER COMMENTS (Post-review additions, clarifications):\n"
        f"{revised_info_text if revised_info_text else 'Not provided.'}"
    )

    print(f"\nDEBUG: Generating main note body...")
    main_note_body_content = generate_neurology_note_body(
        gnb_historical_text,
        gnb_current_visit_context,
        gnb_user_insights
    )

    if "Error generating main note body:" in main_note_body_content:
        return f"FAILED TO GENERATE MAIN NOTE BODY: {main_note_body_content}"
    print("--- END DEBUG: Main Note Body content generated ---")

    diag_historical_context = (
        f"I. BACKGROUND INFORMATION (Prior to this visit):\n"
        f"{background_info_text if background_info_text else 'Not provided.'}\n\n"
        f"II. CURRENT VISIT TRANSCRIPTION:\n"
        f"{transcription_text if transcription_text else 'Not provided.'}"
    )
    diag_user_insights_and_revisions = (
        f"III. USER'S ADDITIONAL INFORMATION/EMPHASIS (Clinician's thoughts, etc.):\n"
        f"{additional_info_text if additional_info_text else 'Not provided.'}\n\n"
        f"IV. USER'S REVISED INFORMATION/FURTHER COMMENTS:\n"
        f"{revised_info_text if revised_info_text else 'Not provided.'}"
    )
    diag_reason_for_visit = "Comprehensive neurological assessment based on all provided information."

    print("\nDEBUG: Running diagnostic assessment...")
    diagnostic_assessment_text = generate_diagnostic_assessment_llm(
        diag_historical_context,
        diag_reason_for_visit,
        diag_user_insights_and_revisions
    )

    if "Error generating diagnostic assessment:" in diagnostic_assessment_text:
        return f"FAILED TO GENERATE DIAGNOSTIC ASSESSMENT: {diagnostic_assessment_text}"
    print("--- END DEBUG: Diagnostic Assessment content generated ---")

    extracted_primary_diagnosis = None
    if diagnostic_assessment_text and "Error generating diagnostic assessment:" not in diagnostic_assessment_text:
        extracted_primary_diagnosis = extract_diagnosis_with_llm(diagnostic_assessment_text)
    print(f"DEBUG: Final extracted_primary_diagnosis for downstream tasks: '{extracted_primary_diagnosis}'")

    final_alz_checklist_text = ""
    checklist_marker_in_main_note = "--- Alzheimer's Disease Candidate Checklist ---"
    raw_checklist_present_in_main_note = False

    raw_template_start_index = main_note_body_content.find(checklist_marker_in_main_note)
    raw_template_end_marker = "--- (End Checklist Template) ---"

    if raw_template_start_index != -1:
        raw_template_end_index = main_note_body_content.find(raw_template_end_marker, raw_template_start_index)
        if raw_template_end_index != -1:
            raw_checklist_present_in_main_note = True
            part_before_template = main_note_body_content[:raw_template_start_index]
            actual_template_block_end = raw_template_end_index + len(raw_template_end_marker)
            part_after_template = main_note_body_content[actual_template_block_end:]
            main_note_body_content = (part_before_template.strip() + "\n\n" + part_after_template.strip()).strip()
            print(
                "DEBUG: Raw Alzheimer's checklist template was initially present and has been removed pending processing.")
        else:
            print(
                "DEBUG: Checklist start marker found, but end marker missing in main_note_body_content. Raw template may not be fully formed or removed correctly.")
            raw_checklist_present_in_main_note = False

    if raw_checklist_present_in_main_note and extracted_primary_diagnosis and is_alzheimers_primary_diagnosis(
            extracted_primary_diagnosis):
        print(
            f"DEBUG: Alzheimer's is considered a primary diagnosis ('{extracted_primary_diagnosis}'). Processing checklist.")
        final_alz_checklist_text = process_alzheimers_checklist_with_llm(
            ALZHEIMERS_CHECKLIST_TEMPLATE_FOR_PROCESSING,
            background_info_text,
            transcription_text,
            f"Additional Info:\n{additional_info_text}\n\nRevised Info:\n{revised_info_text}",
            diagnostic_assessment_text
        )
    elif raw_checklist_present_in_main_note:
        print(
            f"DEBUG: Checklist template was present in initial note, but AD not confirmed as primary by detailed assessment ('{extracted_primary_diagnosis}'). Checklist will be omitted.")
    else:
        print(
            "DEBUG: Alzheimer's checklist template was not included by the initial LLM, or AD is not primary. Checklist will be omitted.")

    patient_criteria_elaboration_text = ""
    generic_criteria_header = "### Supporting Criteria from Diagnostic Algorithm"

    if extracted_primary_diagnosis and is_alzheimers_primary_diagnosis(extracted_primary_diagnosis):
        patient_criteria_elaboration_text = generate_patient_specific_criteria_elaboration(
            extracted_primary_diagnosis,
            background_info_text,
            transcription_text,
            f"Additional Info:\n{additional_info_text}\n\nRevised Info:\n{revised_info_text}",
            diagnostic_assessment_text
        )

    placeholder_med_exp = "[YOUR SEPARATELY GENERATED DETAILED MEDICAL EXPLANATION AND PLAN WILL BE INSERTED HERE BY THE SCRIPT]"
    note_with_medical_explanation = main_note_body_content

    if placeholder_med_exp in note_with_medical_explanation:
        note_with_medical_explanation = note_with_medical_explanation.replace(placeholder_med_exp,
                                                                              diagnostic_assessment_text.strip())
        print(f"DEBUG: Successfully replaced '{placeholder_med_exp}'.")
    else:
        print(f"DEBUG: Placeholder '{placeholder_med_exp}' not found. Appending diagnostic assessment.")
        note_with_medical_explanation = note_with_medical_explanation.strip() + "\n\n" + diagnostic_assessment_text.strip()

    current_note_assembly = note_with_medical_explanation.strip()

    idx_generic_criteria = -1
    if generic_criteria_header:
        idx_generic_criteria = current_note_assembly.find(generic_criteria_header)

    if patient_criteria_elaboration_text:
        if idx_generic_criteria != -1 and generic_criteria_header:
            end_idx_generic = len(current_note_assembly)
            next_heading_match = re.search(r"\n##|\n###", current_note_assembly[idx_generic_criteria + len(
                generic_criteria_header):])  # Simpler regex for next heading
            if next_heading_match:
                end_idx_generic = idx_generic_criteria + len(generic_criteria_header) + next_heading_match.start()
            else:
                double_newline_offset = current_note_assembly[
                                        idx_generic_criteria + len(generic_criteria_header):].find("\n\n")
                if double_newline_offset != -1:
                    end_idx_generic = idx_generic_criteria + len(generic_criteria_header) + double_newline_offset

            part_before_generic = current_note_assembly[:idx_generic_criteria]
            part_after_generic = current_note_assembly[end_idx_generic:]
            current_note_assembly = part_before_generic.strip() + "\n\n" + patient_criteria_elaboration_text.strip() + "\n\n" + part_after_generic.strip()
            print("DEBUG: Replaced generic 'Supporting Criteria' with patient-specific AD criteria elaboration.")
        else:
            current_note_assembly += "\n\n" + patient_criteria_elaboration_text.strip()
            print("DEBUG: Added patient-specific AD criteria elaboration (generic header not found or empty).")
    elif idx_generic_criteria != -1 and generic_criteria_header:
        print("DEBUG: Kept generic 'Supporting Criteria' as AD not primary or specific elaboration not generated.")

    if final_alz_checklist_text:
        current_note_assembly += "\n\n" + final_alz_checklist_text.strip()
        print("DEBUG: Added processed Alzheimer's checklist.")

    literature_summary_section_text = "### Recent Literature Summary\n\nNO INFORMATION FOUND (Diagnosis not extracted or PubMed search failed)"
    if extracted_primary_diagnosis:
        print(f"DEBUG: Proceeding with PubMed search for: {extracted_primary_diagnosis}")
        abstracts = fetch_recent_guidelines(extracted_primary_diagnosis)
        if abstracts:
            literature_summary_section_text = summarize_literature_with_gemini(abstracts, extracted_primary_diagnosis)
        else:
            literature_summary_section_text = f"### Recent Literature Summary\n\nNo recent relevant guidelines found on PubMed for '{extracted_primary_diagnosis}'."
    else:
        print("DEBUG: Skipping PubMed search as no diagnosis was extracted for literature summary.")

    note_with_literature = current_note_assembly.strip() + "\n\n" + literature_summary_section_text.strip()

    missing_info_summary_text = generate_missing_info_summary(note_with_literature)

    final_note = note_with_literature.strip() + \
                 "\n\n" + SIGNATURE.strip() + \
                 "\n\n" + missing_info_summary_text.strip()

    print("DEBUG: Core note processing finished.")
    return final_note.strip()