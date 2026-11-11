"""Rebuild the final IJMI submission package from the authors' original submission zip.

Usage: python tools/build_submission.py <original_IJMI_RheumaPain_submission.zip> <output_folder>

Input: the zip with Manuscript_anonymised_IJMI.docx, Title_page_IJMI.docx, Highlights.docx,
Appendix_A_Supplementary_material.docx, Figure1-4.tif and Supplementary_Data_S1_results.xlsx.
Needs pandoc on the PATH (cover letter, TRIPOD+AI checklist, declaration) and the packages in requirements.txt.

What it changes (everything else is copied unchanged):
  manuscript  authors and affiliations under the title (single-blind review); repository link in the data
              availability statement; AI-assisted code sentence in Section 2.6; reference [8] replaced by
              Tougui et al. 2021 and the four sentences that cited the unpublished thesis reworded
  title page  repository link in "Data and code availability"
  appendix    Figure A.1 replaced by the 600 dpi figure drawn by src/s09_publication_figures.py
  figures     Figures 1-4 saved without alpha channel; Figure_A1.tif and Graphical_abstract.tif drawn from results/
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ZIP, OUT = sys.argv[1], sys.argv[2]
REPO_URL = "https://github.com/Emsh1379/RheumaPain"

MANUSCRIPT_EDITS = [
    ("and high reported accuracy reflects subject-level leakage.",
     "and high window-split accuracy reflects subject-level leakage."),
    ("Its data article reports no baseline models, and an earlier unpublished thesis on these data reported 92% F1 "
     "for pain level [8] without a documented subject-independent evaluation.",
     "Its data article reports no baseline models, so no subject-independent benchmark exists for these data, "
     "although record-wise splits are known to overestimate the performance of clinical machine-learning models "
     "relative to subject-wise evaluation [8]."),
    ("and reproduced the Kaggle results within 0.006 AUC. Reporting follows TRIPOD+AI where applicable [12].",
     "and reproduced the Kaggle results within 0.006 AUC. The analysis code was written and executed with the "
     "assistance of a generative AI tool (Claude, Anthropic) under the authors’ direction; the authors specified "
     "all analyses, reviewed the code and checked every reported number against the saved result files. Reporting "
     "follows TRIPOD+AI where applicable [12]."),
    ("The 92% F1 reported in an earlier thesis on these data [8] is consistent with what we obtained with a random "
     "window split (0.91, binary), and falls to 0.59 when children are held out; the split strategy of that work is "
     "not documented in its public abstract.",
     "Our random window split reached macro-F1 0.91 (binary), the level of accuracy typical of subject-mixed "
     "evaluations, and fell to 0.59 when children were held out, in line with the overestimation of record-wise "
     "relative to subject-wise cross-validation shown for diagnostic applications [8]."),
    ("Previously reported high accuracy reflects subject-level leakage, which inflated macro-F1 from 0.59 to 0.91.",
     "High accuracy under random window splits reflects subject-level leakage, which inflated macro-F1 from 0.59 "
     "to 0.91."),
    ("[8] [Student name to be added], Pain detection in pediatric rheumatology patients with deep learning [MSc "
     "thesis; advisor F. Patlar Akbulut], Istanbul Kültür University, Institute of Graduate Studies, "
     "Istanbul, 2024. Abstract available at: https://enstitu.iku.edu.tr/sites/fbe/files/inline-files/LEE-YL-708.pdf.",
     "[8] I. Tougui, A. Jilbab, J. El Mhamdi, Impact of the choice of cross-validation techniques on the results of "
     "machine learning-based diagnostic applications, Healthc. Inform. Res. 27 (2021) 189–199. "
     "https://doi.org/10.4258/hir.2021.27.3.189."),
    ("openly available in a public repository (link provided on the title page).",
     f"openly available at {REPO_URL}."),
]
TITLE_PAGE_EDITS = [
    ("extracted features and result tables: [GitHub repository URL and Zenodo DOI to be added]",
     f"extracted features and result tables: {REPO_URL}"),
]
AUTHORS = [("Shekoofeh Savabieh", "a,b"), ("Mohammad Emad Sharifi", "c,d,*")]
AFFILIATIONS = [
    ("a", "Children Growth Disorder Research Center, Shahid Sadoughi University of Medical Sciences, Yazd, Iran"),
    ("b", "Department of Pediatrics, Shahid Sadoughi University of Medical Sciences, Yazd, Iran"),
    ("c", "Mother and Newborn Health Research Center, Shahid Sadoughi University of Medical Sciences, Yazd, Iran"),
    ("d", "School of Dentistry, Tehran University of Medical Sciences, Tehran, Iran"),
]
CORRESPONDING = (" Corresponding author: Mohammad Emad Sharifi, No. 11, 6th Alley, Mulla Sadra St., Safaeyeh, "
                 "Yazd 8916863415, Iran. E-mail: emad.sharifi2001@gmail.com")
TITLE_END = "a leakage-aware benchmark and audit of a public dataset</w:t></w:r></w:p>"


def run(text, sup=False, bold=False, ital=False):
    rpr = ("<w:b/>" if bold else "") + ("<w:i/>" if ital else "") + \
          ('<w:vertAlign w:val="superscript"/>' if sup else "")
    return f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{text}</w:t></w:r>'


def para(runs, after=60):
    return ('<w:p><w:pPr><w:pStyle w:val="BodyText"/><w:spacing w:line="360" w:lineRule="auto" '
            f'w:after="{after}"/></w:pPr>{"".join(runs)}</w:p>')


def author_block():
    names = []
    for i, (name, marks) in enumerate(AUTHORS):
        if i:
            names.append(run(", ", bold=True))
        names += [run(name, bold=True), run(marks, sup=True)]
    block = para(names, 120)
    for mark, text in AFFILIATIONS:
        block += para([run(mark, sup=True), run(" " + text, ital=True)])
    return block + para([run("*", sup=True), run(CORRESPONDING)], 240)


def rewrite_docx(src, dst, edits=(), insert_after=None, media=None):
    """Copy a .docx, replacing text in word/document.xml (each old text must occur exactly once)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item)
            if item.filename == "word/document.xml":
                xml = data.decode()
                for old, new in edits:
                    assert xml.count(old) == 1, f"{os.path.basename(src)}: expected one match for: {old[:70]}"
                    xml = xml.replace(old, new)
                if insert_after:
                    marker, block = insert_after
                    k = xml.index(marker) + len(marker)
                    xml = xml[:k] + block + xml[k:]
                data = xml.encode()
            if media and item.filename in media:
                data = media[item.filename]
            zout.writestr(item, data)


def main():
    os.makedirs(OUT, exist_ok=True)
    work = tempfile.mkdtemp()
    orig, figs = os.path.join(work, "orig"), os.path.join(work, "figs")
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall(orig)
    py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "src", "s09_publication_figures.py"),
                    os.path.join(ROOT, "results"), figs], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([py, os.path.join(ROOT, "src", "s10_graphical_abstract.py"),
                    os.path.join(figs, "Graphical_abstract")], check=True, stdout=subprocess.DEVNULL)

    rewrite_docx(os.path.join(orig, "Manuscript_anonymised_IJMI.docx"), os.path.join(OUT, "Manuscript_IJMI.docx"),
                 MANUSCRIPT_EDITS, insert_after=(TITLE_END, author_block()))
    rewrite_docx(os.path.join(orig, "Title_page_IJMI.docx"), os.path.join(OUT, "Title_page_IJMI.docx"),
                 TITLE_PAGE_EDITS)

    a1 = Image.open(os.path.join(figs, "FigureA1.tif"))
    buf = io.BytesIO()
    a1.save(buf, "PNG", dpi=(600, 600), optimize=True)
    old_extent = 'cx="5669280" cy="2418893"'
    new_extent = f'cx="5669280" cy="{round(5669280 * a1.height / a1.width)}"'
    appendix = os.path.join(orig, "Appendix_A_Supplementary_material.docx")
    with zipfile.ZipFile(appendix) as z:
        n = z.read("word/document.xml").decode().count(old_extent)
    tmp = os.path.join(work, "appendix.docx")
    with zipfile.ZipFile(appendix) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item)
            if item.filename == "word/document.xml":
                assert n >= 1
                data = data.decode().replace(old_extent, new_extent).encode()
            if item.filename == "word/media/image1.png":
                data = buf.getvalue()
            zout.writestr(item, data)
    shutil.copy(tmp, os.path.join(OUT, "Appendix_A_Supplementary_material.docx"))

    for i in range(1, 5):  # Elsevier asks for figures without an alpha channel
        Image.open(os.path.join(orig, f"Figure{i}.tif")).convert("RGB").save(
            os.path.join(OUT, f"Figure{i}.tif"), dpi=(600, 600), compression="tiff_lzw")
    shutil.copy(os.path.join(figs, "FigureA1.tif"), os.path.join(OUT, "Figure_A1.tif"))
    shutil.copy(os.path.join(figs, "Graphical_abstract.tif"), os.path.join(OUT, "Graphical_abstract.tif"))
    for name in ("Highlights.docx", "Supplementary_Data_S1_results.xlsx"):
        shutil.copy(os.path.join(orig, name), os.path.join(OUT, name))
    sources = os.path.join(ROOT, "tools", "submission_sources")
    for name in ("Cover_letter", "TRIPOD_AI_checklist", "Declaration_of_competing_interests"):
        subprocess.run(["pandoc", os.path.join(sources, name + ".md"), "-o", os.path.join(OUT, name + ".docx")],
                       check=True)
    shutil.rmtree(work)
    print("\n".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
