**SKYNET (SCOP K-mer Hybrid Network)**
This repository represent SCOP, a hybdrid interpretable ML model to predict protein structural class from sequence.

🚀 **Folder structure**
- Dataset folder contains all the datasets
- models folder contains trained models
- codes folder contains necessary codes

**Dependencies**
- numpy = 1.19.5
- scikit-learn = 0.24.2
- tensorflow = 2.6.2
- keras = 2.6.0

🧬 **Example Usage**

[1] Enter the protein sequence (Only sequence, without headers):
MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLAAG

[2] Predicted structural class and top important features:
    - alpha helix 2: beta-sheet 3: alpha/beta 4. alpha+beta
    - Top 5 important 6-mers based on Random Forest's feature importances:
    6-mer: adkelk with Importance: 0.040147
    6-mer: gvdaln with Importance: 0.028921
    6-mer: nklqag with Importance: 0.027508
    6-mer: kflvvd with Importance: 0.026918
    6-mer: dkelkf with Importance: 0.025589
    6-mer: iradga with Importance: 0.022588

[3] Press q to quit or enter a new sequence
