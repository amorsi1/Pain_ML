# Pain_ML

This codebase integrates outputs of unsupervised and supervised machine learning techniques to create a multi-modal dataset for analyzing videos of freely moving mice.

These outputs come from:
* [BlackBox Analysis](https://github.com/blackbox-bio/analysis-public): Extracts paw pressure for each frame. Custom tools to identify paw guarding and rearing are not in this codebase, but reflected in the features.h5 files
* [DeepEthogram](https://github.com/jbohnslav/deepethogram/tree/master): CNN to identify paw licking/biting, paw shaking, and face grooming
* [Keypoint-Moseq](https://github.com/dattalab/keypoint-moseq): Extracts syllables that reflect the underlying structure of behavior. The upregulation of certain syllables is reflective of pain (with the correct controls), as well as changes in the sequences of how these syllables are used

The performance of these approaches in discriminating between pain states can be analyzed by the methodology presented in:

Wiltschko AB, Tsukahara T, Zeine A, Anyoha R, Gillis WF, Markowitz JE, Peterson RE, Katon J, Johnson MJ, Datta SR. Revealing the structure of pharmacobehavioral space through motion sequencing. Nat Neurosci. 2020 Nov;23(11):1433-1443. doi: 10.1038/s41593-020-00706-3. Epub 2020 Sep 21. PMID: 32958923; PMCID: PMC7606807.



