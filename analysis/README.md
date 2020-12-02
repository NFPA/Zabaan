# Details

## alignment_viz.ipynb 
Notebook to generate Alignment plots from the alignment files.
Inputs: 
- Processed English and Spanish Files
- Alignment file generated using FastAlign

Output:
- Visualization Plot

## phrase_table.ipynb

Generates DataFrame like phrase table from the alignment file.

## vocab_analysis.ipynb

Provided in depth analysis of unique tokens disctinct with NFPA data sets and compares overlap with tokens available from open source datasets. 
This helps in inital troubleshooting of errors during translation especially for Out-Of-Vocabulary (OOV) tokens. 

## vocab_detection_func.ipynb
Function to detect if a bunch of tokens are present in NFPA, General or shared vocabulary list.