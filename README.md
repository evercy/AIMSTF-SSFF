# AIMSTF-SSFF

The PyTorch implementation for the paper 'Bimodal Emotion Recognition Based on Adaptive Integration of Multi-level Spatial-Temporal Features and Specific-Shared Feature Fusion'.

# Overview
![image](https://github.com/evercy/AIMSTF_SSFF/blob/main/Figure/%E5%9B%BE1.jpg)
<p align="center">Architecture of the AIMSTF-SSFF </p>

Abstract: 
There are usually two challenging issues in the field of bimodal emotion recognition combining electroencephalogram(EEG) and facial images: (1) How to learn more significant emotionally semantic features from EEG signals in an end-to-end manner; (2) How to effectively integrate bimodal information to capture the coherence and complementarity of emotional semantics among bimodal features. In this paper, one bimodal emotion recognition model is proposed via the adaptive integration of multi-level spatial-temporal features and the fusion of specific-shared features. On the one hand, in order to obtain more significant emotionally semantic features from EEG signals, one module, called adaptive integration of multi-level spatial-temporal features, is designed. The spatial-temporal features of EEG signals are firstly captured with a dual-flow structure before the features from each level are integrated by taking into consideration the weights deriving from the similarity of features. Finally, the relatively important feature information from each level is adaptively learned based on the gating mechanism. On the other hand, in order to leverage the emotionally semantic consistency and complementarity between EEG signals and facial images, one module fusing specific-shared features is devised. Emotionally semantic features are learned jointly through two branches: specific-feature learning and shared-feature learning. The loss function is also incorporated to automatically extract the specific semantic information for each modality and the shared semantic information among the modalities. On both the DEAP and MAHNOB-HCI datasets, cross-experimental verification and 5-fold cross-validation strategies are used to assess the performance of the proposed model. The experimental results and their analysis demonstrate that the model achieves competitive results, providing an effective solution for bimodal emotion recognition based on EEG signals and facial images.

# Dependencies
+ Python 3.6
+ PyTorch 1.7.1
+ torchvision 0.8.2

# Data availability
All datasets are freely available in public repositories. 

+ DEAP: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
+ MAHNOB-HCI: https://mahnob-db.eu/hci-tagging/


# Contact
If you have any questions, please feel free to reach me out at qsun@xaut.edu.cn and 2210321182@stu.xaut.edu.cn.

# Acknowledgements
This project is built upon [DeepVANet](https://github.com/geekdanielz/DeepVANet) and [TSception](https://github.com/yi-ding-cs/TSception). Thanks for their great codebase.

# Citation
If this work is useful to you, please cite:

> Sun, Q., Chen, Y.: Bimodal Emotion Recognition Based on Adaptive Integration of Multi-level Spatial-Temporal Features and Specific-Shared Feature Fusion[J]. Journal of Electronics & Information Technology, 46(2):1-14, 2024. doi: 10.11999/JEIT231110.

Qiang Sun<sup>1,2</sup>,Yuan Chen<sup>1</sup>
1.	Department of Communication Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
2.	Xi’an Key Laboratory of Wireless Optical Communication and Network Research, Xi’an 710048, China

