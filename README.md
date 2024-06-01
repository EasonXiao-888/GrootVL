# 🌲 GrootVL: Tree Topology is All You Need in State Space Model
[Yicheng Xiao<sup><span>1,*</span></sup>](https://easonxiao-888.github.io/), [Lin Song<sup><span>2,📧,*</span></sup>](https://linsong.info/), [Shaoli Huang<sup>2</sup>](https://scholar.google.com/citations?user=o31BPFsAAAAJ&hl=en&oi=ao), [Jiangshan Wang<sup><span>1</span></sup>](https://scholar.google.com/citations?hl=en&user=HoKoCv0AAAAJ), [Siyu Song<sup><span>3</span></sup>](), [Yixiao Ge<sup><span>2</span></sup>](http://geyixiao.com/), [Xiu Li<sup><span>1,📧</span></sup>](https://www.sigs.tsinghua.edu.cn/lx/main.htm) and [Ying Shan<sup><span>2</span></sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)

\* Equal contribution  📧 Corresponding author

<sup>1</sup> Tsinghua University, <sup>2</sup> Tencent AI Lab, <sup>3</sup> South China Normal University

## 📖 Abstract
The state space models, employing recursively propagated features, demonstrate strong representation capabilities comparable to Transformer models and superior efficiency. However, constrained by the inherent geometric constraints of sequences, it still falls short in modeling long-range dependencies. To address this issue, we propose the GrootVL network, which first dynamically generates a tree topology based on spatial relationships and input features. Then, feature propagation is performed based on this graph, thereby breaking the original sequence constraints to achieve stronger representation capabilities. Additionally, we introduce a linear complexity dynamic programming algorithm to enhance long-range interactions without increasing computational cost. GrootVL is a versatile multimodal framework that can be applied to both visual and textual tasks. Extensive experiments demonstrate that our method significantly outperforms existing structured state space models on image classification, object detection and segmentation. Besides, by fine-tuning large language models, our approach achieves consistent improvements in multiple textual tasks at minor training cost.

---

<p align="center">
 <img src="assets/tree_ssm.png" width="100%">
</p>

## 🔨 Model Zoo

#### Vision Tasks
<details>
<summary> ImageNet-1k Image Classification </summary>
<br>

<div>

|      name      |   pretrain   | resolution | acc@1 | #param | FLOPs |                                                                             download                                                                              |
| :------------: | :----------: | :--------: | :---: | :----: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GrootV-T  | ImageNet-1K  |  224x224   | 83.4  |  30M   |  4.8G   |       [ckpt]() \| [cfg]()       |
| GrootV-S  | ImageNet-1K  |  224x224   | 84.2  |  51M   |  8.5G   |       [ckpt]() \| [cfg]()       |
| GrootV-B  | ImageNet-1K  |  224x224   | 84.8  |  91M   |  15.1G  |       [ckpt]() \| [cfg]()       |
| GrootV-L  | ImageNet-22K |  384x384   | RUNNING  |  -  | -  |  [ckpt]() \| [cfg]()  |
</div>

</details>

<details>
<summary> COCO Object Detection and Instance Segmentation </summary>
<br>
<div>

|    backbone    |   method   | schedule  | box mAP | mask mAP | #param | FLOPs |                                                                                     download                                                                                      |
| :------------: | :--------: | :---: | :-----: | :------: | :----: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GrootV-T  | Mask R-CNN |  1x   |  47.0   |   42.7   |  49M   | 265G  | [ckpt]() \| [cfg]() |
| GrootV-T  | Mask R-CNN |  3x   |  49.0   |   43.8   |  49M   | 265G  | [ckpt]() \| [cfg]() |
| GrootV-S  | Mask R-CNN |  1x   |  48.6   |   43.6   |  70M   | 341G  | [ckpt]() \| [cfg]() |
| GrootV-S  | Mask R-CNN |  3x   |  50.1   |   44.6   |  70M   | 341G  | [ckpt]() \| [cfg]() |

</div>

</details>

<details>
<summary> ADE20K Semantic Segmentation </summary>
<br>
<div>

|    backbone    |   method    | resolution | mIoU (ss/ms) | #param | FLOPs |                                                                                           download                                                                                           |
| :------------: | :---------: | :--------: | :----------: | :----: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GrootV-T  |   UperNet   |  512x512   | 48.5 / 49.4  |  60M   | 941G  |  [ckpt]() \| [cfg]()  |
| GrootV-S  |   UperNet   |  512x512   | 50.7 / 51.7  |  82M   | 1019G |  [ckpt]() \| [cfg]()  |
</div>
</details>


#### Language Tasks
<details>
<summary> Language Understanding </summary>
<br>

<div>

|      Method      |   PIQA &uarr;  | Arc-E &uarr; | sst &uarr; | WinoGrande &uarr; | LAMBADA-ppl &darr; |  race &uarr; | Openbookqa &uarr; | Average Acc &uarr; | download |
| :------------: | :----------: | :--------: | :---: | :----: | :---: | :---: | :---: | :---: |:---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mamba  | 64.5  |  48.0   | 65.6  |  51.8   |  16.1  | 27.4 | 16.8 | 45/7 | [model]() |
| + LoRA | 64.7 | 48.3 | 65.1 | 52.2 | 17.7 | 28.6 | 17.8 | 46.1| [model]() |
| + GrootL  | 65.0 | 49.8 | 69.5 | 51.1 | 15.9 | 28.9 | 19.2 | 47.2| [model]() |
</div>
</details>