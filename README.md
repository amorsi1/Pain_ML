This codebase integrates outputs of unsupervised and supervised machine learning techniques to create a multi-modal dataset for analyzing videos of freely moving mice.

These outputs come from:
* [BlackBox Analysis](https://github.com/blackbox-bio/analysis-public): Extracts paw pressure for each frame. Custom tools to identify paw guarding and rearing are not in this codebase, but reflected in the features.h5 files
* [DeepEthogram](https://github.com/jbohnslav/deepethogram/tree/master): CNN to identify paw licking/biting, paw shaking, and face grooming
* [ARBEL](https://www.biorxiv.org/content/10.1101/2024.12.01.625907v1) XGBoost-based classifier that uses tracking data only 
* [Keypoint-Moseq](https://github.com/dattalab/keypoint-moseq): Extracts syllables that reflect the underlying structure of behavior. The upregulation of certain syllables is reflective of pain (with the correct controls), as well as changes in the sequences of how these syllables are used

Altogether these can be composed to get a more holistic understanding of the mouse's behavior:
![raster_combined_short](https://github.com/user-attachments/assets/9d5e906f-dced-4747-840e-8e692e9a4cf5)


When framed as a classification problem, the tradeoffs of each approach become more clear :
<img width="941" height="313" alt="image" src="https://github.com/user-attachments/assets/73dbc566-81f8-4dd4-b0d8-23d2878b44a4" />



