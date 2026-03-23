# =========================
# IMPORT LIBRARIES
# =========================

from pathlib import Path                         # to check if file exists
from pypdf import PdfReader                      # to read PDF files
from sentence_transformers import SentenceTransformer  # local embedding model
from sklearn.metrics.pairwise import cosine_similarity # to compare similarity


print("Program started")


# =========================
# SETTINGS (CONFIG)
# =========================

PDF_FILE = "sample.pdf"      # name of your PDF file
CHUNK_SIZE = 1200            # size of each text chunk
CHUNK_OVERLAP = 200          # overlap between chunks
TOP_K = 3                    # number of results to return


# =========================
# FUNCTION: READ PDF
# =========================

def read_pdf_text(pdf_path):
    """
    Reads the PDF and returns all text as one string
    """
    reader = PdfReader(pdf_path)   # open PDF
    pages = []

    # go through each page and extract text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    # combine all pages into one long string
    return "\n".join(pages)


# =========================
# FUNCTION: SPLIT TEXT INTO CHUNKS
# =========================

def chunk_text(text, chunk_size=1200, overlap=200):
    """
    Splits large text into smaller overlapping chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size           # define chunk end
        chunk = text[start:end].strip()    # extract chunk

        if chunk:
            chunks.append(chunk)

        # move forward but keep overlap for context
        start += chunk_size - overlap

    return chunks


# =========================
# MAIN PROGRAM
# =========================

def main():

    # --- Check if PDF exists ---
    print("Checking PDF file...")
    if not Path(PDF_FILE).exists():
        raise FileNotFoundError(f"Could not find {PDF_FILE}")

    # --- Read PDF ---
    print(f"Reading {PDF_FILE}...")
    full_text = read_pdf_text(PDF_FILE)

    if not full_text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    # --- Split text into chunks ---
    print("Splitting text into chunks...")
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # --- Load local embedding model ---
    print("Loading local embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Convert chunks into embeddings ---
    print("Creating embeddings for PDF chunks...")
    chunk_embeddings = model.encode(chunks)

    # --- Ask user a question ---
    question = input("\nAsk a question about the PDF: ").strip()
    if not question:
        raise ValueError("You need to type a question.")

    # --- Convert question into embedding ---
    print("Creating embedding for your question...")
    question_embedding = model.encode([question])

    # --- Compare question with all chunks ---
    print("Finding most relevant chunks...\n")
    scores = cosine_similarity(question_embedding, chunk_embeddings)[0]

    # --- Get top K most relevant chunks ---
    top_indices = scores.argsort()[::-1][:TOP_K]

    # --- Display results ---
    print("Most relevant parts of the PDF:\n")

    print("\nSummary based on most relevant sections:\n")

    combined_text = " ".join([chunks[i] for i in top_indices])

    # simple "manual" summary (no AI)
    sentences = combined_text.split(".")
    summary = ". ".join(sentences[:5])  # take first 5 sentences

    print(summary)

# =========================
# RUN PROGRAM
# =========================

if __name__ == "__main__":
    main()