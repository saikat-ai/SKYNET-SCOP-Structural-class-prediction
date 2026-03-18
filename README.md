**SKYNET (SCOP K-mer Hybrid Network)**
This repository represent SCOP, a hybdrid interpretable ML model to predict protein structural class from sequence.

🚀 **Folder structure**
- Dataset folder contains all the datasets
- models folder contains trained models
- codes folder contains necessary codes

**Dependencies**
numpy = 1.19.5
scikit-learn = 0.24.2
tensorflow = 2.6.2
keras = 2.6.0

🧬 **Example Usage**
[1] Enter the protein sequence (Only sequence, without headers):
MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLAAG

[2] Predicted structural class and top important features:
    1: alpha helix 2: beta-sheet 3: alpha/beta 4. alpha+beta
    Top 20 importat 6-mer features are shown

[3] Do you want to input another sequence? (yes/no):
if yes enter another sequence in same way and if no the program will be ended.
