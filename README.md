# AIMSTF-SSFF
这是对论文“多层次时空特征自适应集成与特有-共享特征融合的双模态情感识别”的PyTorch实现。
The PyTorch implementation for the paper 'Bimodal Emotion Recognition Based on Adaptive Integration of Multi-level Spatial-Temporal Features and Specific-Shared Feature Fusion'.

孙强<sup>1,2</sup>, 陈远<sup>1</sup>
1. 西安理工大学自动化与信息工程学院通信工程系 西安 710048 中国 
2. 西安市无线光通信与网络研究重点实验室   西安  710048 中国 

Qiang Sun<sup>1,2</sup>,Yuan Chen<sup>1</sup>
1.	Department of Communication Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
2.	Xi’an Key Laboratory of Wireless Optical Communication and Network Research, Xi’an 710048, China

# 概述（Overview）
![image](https://github.com/evercy/AIMSTF_SSFF/blob/main/Figure/%E5%9B%BE1.jpg)
<p align="center">AIMSTF-SSFF的结构示意图 </p>
<p align="center">Architecture of the AIMSTF-SSFF </p>

摘要： 在结合脑电(EEG)信号与人脸图像的双模态情感识别领域中，通常存在两个挑战性问题：(1)如何从EEG信号中以端到端方式学习到更具显著性的情感语义特征；(2)如何充分利用双模态信息，捕捉双模态特征中情感语义的一致性与互补性。为此，提出了多层次时空特征自适应集成与特有-共享特征融合的双模态情感识别模型。一方面，为从EEG信号中获得更具显著性的情感语义特征，设计了多层次时空特征自适应集成模块。该模块首先通过双流结构捕捉EEG信号的时空特征，再通过特征相似度加权并集成各层次的特征，最后利用门控机制自适应地学习各层次相对重要的情感特征。另一方面，为挖掘EEG信号与人脸图像之间的情感语义一致性与互补性，设计了特有-共享特征融合模块，通过特有特征的学习和共享特征的学习来联合学习情感语义特征，并结合损失函数实现各模态特有语义信息和模态间共享语义信息的自动提取。在DEAP和MAHNOB-HCI两种数据集上，采用跨实验验证和5折交叉验证两种实验手段验证了提出模型的性能。实验结果表明，该模型取得了具有竞争力的结果，为基于EEG信号与人脸图像的双模态情感识别提供了一种有效的解决方案。
Abstract: There are usually two challenging issues in the field of bimodal emotion recognition combining electroencephalogram(EEG) and facial images: (1) How to learn more significant emotionally semantic features from EEG signals in an end-to-end manner; (2) How to effectively integrate bimodal information to capture the coherence and complementarity of emotional semantics among bimodal features. In this paper, one bimodal emotion recognition model is proposed via the adaptive integration of multi-level spatial-temporal features and the fusion of specific-shared features. On the one hand, in order to obtain more significant emotionally semantic features from EEG signals, one module, called adaptive integration of multi-level spatial-temporal features, is designed. The spatial-temporal features of EEG signals are firstly captured with a dual-flow structure before the features from each level are integrated by taking into consideration the weights deriving from the similarity of features. Finally, the relatively important feature information from each level is adaptively learned based on the gating mechanism. On the other hand, in order to leverage the emotionally semantic consistency and complementarity between EEG signals and facial images, one module fusing specific-shared features is devised. Emotionally semantic features are learned jointly through two branches: specific-feature learning and shared-feature learning. The loss function is also incorporated to automatically extract the specific semantic information for each modality and the shared semantic information among the modalities. On both the DEAP and MAHNOB-HCI datasets, cross-experimental verification and 5-fold cross-validation strategies are used to assess the performance of the proposed model. The experimental results and their analysis demonstrate that the model achieves competitive results, providing an effective solution for bimodal emotion recognition based on EEG signals and facial images.

# 依赖项（Dependencies）
+ Python 3.6
+ PyTorch 1.7.1
+ torchvision 0.8.2

# 数据可用性（Data availability）
所有数据集都可以在公共数据库中免费获得。
All datasets are freely available in public repositories. 
+ DEAP: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
+ MAHNOB-HCI: https://mahnob-db.eu/hci-tagging/


# 联系（Contact）
如果您有任何问题，请随时通过qsun@xaut.edu.cn和2210321182@stu.xaut.edu.cn与我们联系。
If you have any questions, please feel free to reach me out at qsun@xaut.edu.cn and 2210321182@stu.xaut.edu.cn.

# 致谢（Acknowledgements）
This project is built upon [DeepVANet](https://github.com/geekdanielz/DeepVANet) and [TSception](https://github.com/yi-ding-cs/TSception). Thanks for their great codebase.

# 引用（Citation）
如果本论文对您有所帮助，请注明出处:
孙强, 陈远. 多层次时空特征自适应集成与特有-共享特征融合的双模态情感识别[J]. 电子与信息学报. doi: 10.11999/JEIT231110
If this work is useful to you, please cite:
> Sun, Q., Chen, Y.: Bimodal Emotion Recognition Based on Adaptive Integration of Multi-level Spatial-Temporal Features and Specific-Shared Feature Fusion[J]. Journal of Electronics & Information Technology.  doi: 10.11999/JEIT231110.

