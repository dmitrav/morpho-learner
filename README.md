# Comparing representations of biological data learned with different AI paradigms, augmenting and cropping strategies

## Abstract

Recent advances in computer vision and robotics enabled automated large-scale biological image analysis. Various machine learning approaches have been successfully applied to phenotypic profiling. However, it remains unclear how they compare in terms of biological feature extraction. In this study, we propose a simple CNN architecture and implement weakly-supervised, self-supervised, unsupervised and regularized learning of image representations. We train 16 deep learning setups on the 770k cancer cell images dataset under identical conditions, using different augmenting and cropping strategies. We compare the learned representations by evaluating multiple metrics for each of three downstream tasks: i) distance-based similarity analysis of known drugs, ii) classification of drugs versus controls, iii) clustering within cell lines. We also compare training times and memory usage. Among all tested setups, multi-crops and random augmentations generally improved performance across tasks, as expected. Strikingly, self-supervised models showed competitive performance being up to 11 times faster to train. Regularized learning required the most of memory and computation to deliver arguably the most informative features. We observe that no single combination of augmenting and cropping strategies consistently resulted in top performance across tasks and recommend prospective research directions.

![Model architectures](https://github.com/dmitrav/morpho-learner/blob/master/img/methods.png)

## Full text

Available [here](https://openreview.net/pdf?id=RPR7hjLYTyU).
