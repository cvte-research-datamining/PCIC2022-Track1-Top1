# PCIC2022 Causal Inference and Transfer Learning

[PCIC赛道1比赛地址](https://competition.huaweicloud.com/information/1000041792/introduction?track=107)

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E7%BA%BF%E4%B8%8A%E5%88%86%E6%95%B0%E6%8E%92%E5%90%8D.png" width = 60%/> </div>


## 代码目录

```
├── 01-dataset 数据集1
│   ├── main.sh 入口
│   ├── README.md
│   ├── requirements.txt
│   └── src
│       ├── conv
│       │   ├── conv.py
│       │   └── run.sh
│       ├── finetune
│       │   ├── finetune.py
│       │   ├── pretrain.py
│       │   └── run.sh
│       ├── fusion
│       │   ├── fusion.py
│       │   └── run.sh
│       ├── input
│       ├── lstm
│       │   ├── lstm.py
│       │   └── run.sh
│       └── process
│           ├── process.py
│           └── run.sh
├── 02-03-dataset 数据集2和3
│   ├── main.sh 入口
│   ├── README.md
│   ├── requirements.txt
│   └── src
│       ├── 01-process
│       │   ├── np2df.py
│       │   └── run.sh
│       ├── 02-data
│       │   ├── finetune
│       │   │   ├── finetune.py
│       │   │   └── run.sh
│       │   └── lstm
│       │       ├── lstm.py
│       │       └── run.sh
│       ├── 03-data
│       │   ├── 01-feat
│       │   │   ├── feat.py
│       │   │   ├── run.sh
│       │   │   └── sensor_process.py
│       │   ├── 02-catboost
│       │   │   ├── graft
│       │   │   │   ├── cat_graft.py
│       │   │   │   └── run.sh
│       │   │   └── kfold
│       │   │       ├── cat_kfold.py
│       │   │       └── run.sh
│       │   └── 03-lightgbm
│       │       └── 01-lgb-graft
│       │           ├── lgb_graft.py
│       │           └── run.sh
│       └── 04-fusion
│           ├── fusion.py
│           └── run.sh
├── figs
│   ├── conv.svg
│   ├── dataset12分数.png
│   ├── dataset3分数.png
│   ├── K折交叉验证.svg
│   ├── lstm.svg
│   ├── step分布.png
│   ├── 嫁接学习.svg
│   ├── 密度分布.png
│   ├── 故障检测.svg
│   ├── 标签分布.png
│   ├── 特征工程.svg
│   ├── 特征重要性.png
│   ├── 线上分数排名.png
│   └── 线下线上分数.png
├── PCIC2022_Track1_Top1_CVTEDMer.pptx 答辩ppt
├── README.md
└── requirements.txt
```


## 复现

大赛共提供了三份数据集，分为AB榜两个阶段，数据1作为A榜阶段，数据集2、3一起提交作为B榜阶段

1. 数据集1，在 `01-dataset` 目录下运行 `sh run.sh` 即可复现数据集1的结果

2. 数据集2、3，在 `02-03-dataset` 目录下运行 `sh run.sh` 即可复现数据集2、3的结果

## 任务介绍

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E6%95%85%E9%9A%9C%E6%A3%80%E6%B5%8B.svg" width = 60%/> </div>

1. 本次任务为家庭宽带网络中的故障预测任务（二分类），选手需建立一个基于迁移学习的故障预测模型对未来7天是否会发生故障做出预测。

2. 数据方面，大赛共提供了数据结构相同的三份数据集，其中数据集1和2是通过模拟生成的数据，数据集3是真实场景下采集得到的数据。每份数据集中又分别包含了两个城市的数据，城市A和城市B，其中城市A中的数据都带有标签，城市B中的20%的数据带有标签，城市B中的80%的数据没有标签，为待预测数据，举办方希望通过对城市A的大量数据建立迁移学习方案对城市B的数据做出预测；每个样本由两部分组成，第一部分为历史7天的多变量时间序列，共有10个测量值，测量频率为15分钟，故维度为672*10；第二部分为样本标签，每个样本的标签记录为其未来7天内是否发生故障。评价指标为F1-score。

## 任务理解

### 标签分布

本任务是故障预测中的二分类任务，如图所示为三份数据集的标签分布情况，可以发现：
- 城市A与B中的标签分布并不一致，数据集1和2中的城市A的黑样本比例为10%左右，而城市B中的黑样本比例为15%左右，数据集3中的城市A的黑样本比例为10%左右，而城市B中的黑样本比例为5.7%左右。因此可考虑微调预训练模型，具体地，在城市A中的样本上预训练一个模型，然后在城市B中的20%的数据上做微调；
- 数据集1和2的标签分布较一致，而与数据集3的标签分布差异较大，因而可以考虑，数据集1和2使用相同的模型，而数据集3则需要考虑使用其他的模型。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E6%A0%87%E7%AD%BE%E5%88%86%E5%B8%83.png" width = 60%/> </div>

### 数据分布

1. 如图所示为三份数据集10个测量值的密度分布图，可以看出：
- 数据集1和2分布基本一致，与数据集3的分布有着极大的不同，同样地，可以考虑数据集1和2使用相同的模型，而数据集3则需要使用其他的模型；
- 每份数据集内部中的城市A、城市B和城市B-test分布较为一致。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E5%AF%86%E5%BA%A6%E5%88%86%E5%B8%83.png" width = 60%/> </div>

2. 如图所示为黑白样本在三份数据的672个步长上的分布，上面两幅子图分别为数据集1和数据集2的分布，下面的子图为数据集3的分布，可以发现：
- 在数据集1和2中，标签为1的数据在672个步长上波动情况要大于标签为0的数据，可以考虑用shift、diff、rolling等特征；
- 在数据集3中，标签为1和0的样本在某些测量值上有较好的区分度，比如sensor_0、sensor_2、sensor_4、sensor_5等，特征工程可以考虑，计算数据在不同步长上的均值，然后用step上的值减去均值，进一步地，可以判断step上的值减去均值是否大于0。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/step%E5%88%86%E5%B8%83.png" width = 60%/> </div>

## 解决方案

### 线下验证

为充分利用所有数据和获得稳定的线下验证结果，本方案采用了如图所示的K折交叉验证的方法。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/K%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.svg" width = 60%/> </div>

如图所示为线上测试分数和线下验证分数的对比结果，可以发现两者有着较强的相关关系，说明该线下验证方案是稳定可靠的。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E7%BA%BF%E4%B8%8B%E7%BA%BF%E4%B8%8A%E5%88%86%E6%95%B0.png" width = 60%/> </div>

### 模型LSTM

LSTM是设计用来处理时序数据的，可以较好地解决长序列训练过程中的梯度消失和梯度爆炸问题，因而我们团队设计了如图所示的基于LSTM的残差连接模型。
- 特征工程，上述任务理解有介绍，包括shift、diff、rolling等特征。
- 迁移学习：① 数据集1中城市A和B的数据，标签分布不一致，可以在城市A的数据上做预训练，然后在城市B上的数据做微调；② 也可在城市A+城市B的数据上做预训练，然后只在城市B上的数据做微调。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/lstm.svg" width = 60%/> </div>

### 模型Conv1d

卷积神经网络最初是设计用来处理图像数据的，后来设计出了一维卷积神经网络用来处理时序数据，因而我们团队也设计了如图所示的Conv1d模型。
- 特征工程和迁移学习与LSTM模型基本一致。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/conv.svg" width = 60%/> </div>

### 数据集1和2结果

1. 数据集1中，预训练方式为在城市A+B的数据上预训练，然后在城市B的数据上做微调，另外预训练+微调的结果稍弱于直接在城市A+B的数据上训练的结果，最终数据集1的线上分数为0.9334。

2. 数据集2中，预训练方式为在城市A的数据上预训练，然后在城市B的数据上做微调，预训练+微调的结果好于直接训练的结果。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/dataset12%E5%88%86%E6%95%B0.png" width = 60%/> </div>

### 嫁接学习

1. 嫁接学习，是一种迁移学习，一开始是用来描述将一个树模型的输出作为另一个树模型的输入的方法，此种方法与树的繁殖中的嫁接类似，故而得名，特别适用于数据分布不一致的场景中。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E5%AB%81%E6%8E%A5%E5%AD%A6%E4%B9%A0.svg" width = 60%/> </div>

2. 如图10所示为嫁接学习的流程，这里需要三份数据（城市B和城市B-test同分布，但城市A与前两者不同分布），
- **第1阶段**：模型1在城市A上训练，然后在城市B和城市B-test上预测，得到预测结果pred2和pred3，将其添加到原始列的后面，命名为pred；
- **第2阶段**：模型2在城市B上训练，然后在城市B-test上预测，得到pred4，作为最后的结果，这里的城市B和城市B-test与原始的数据不同之处在于多了一列pred，是由城市A迁移学习得到的。

3. 进一步优化（如果只有两份数据，城市A和B），
- **第1阶段**：在城市A上做K折交叉训练，得到每一折的预测结果，并添加到城市A原始列的后面，命名为pred，同时对城市B预测，得到K个结果，取平均，同样地，添加到城市B原始列的后面，命名为pred；
- **第2阶段**：在添加了pred列的城市A上训练，对城市B做预测，得到最终结果。

### 特征工程

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B.svg" width = 60%/> </div>

数据集3是真实场景下采集到的数据，与前两个数据集有很大的不同，可以在吸收前两个数据集所使用的特征下，做进一步的特征开发，特征设计角度如上图所示。任务理解部分提到，数据集3中的正负样本在不同step上有着明显的区分度，需重点开发在不同step上的特征。

### 数据集3结果

1. 在数据集3中，嫁接方式为优化后的方式（K折交叉），如图12所示为数据集3的线下验证情况，可以看出：Catboost的结果优于LightGBM的结果，并且两者使用嫁接学习的结果均较直接训练，有明显的提升，Catboost+嫁接学习有8个千的提升，LightGBM+嫁接学习有5个千的提升。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/dataset3%E5%88%86%E6%95%B0.png" width = 60%/> </div>

2. 从特征重要性来看，嫁接学习的特征（pred）也远高于其他特征，说明该方案是非常稳定可靠的。

<div align=center> <img src="https://github.com/cvte-research-datamining/PCIC2022-Track1-Top1/blob/master/figs/%E7%89%B9%E5%BE%81%E9%87%8D%E8%A6%81%E6%80%A7.png" width = 60%/> </div>

## 总结与展望

1. 数据>特征工程>模型，在比赛数据不能扩展的情况下，对数据的理解做的特征工程能够有效地提升分数。然而，在看了其他选手的解决方案时，发现数据分析的过程偏少，更多的是直接上模型，而评委也比较看重模型，导致被翻盘；

2. 对于三份不同的数据，设计了三种不同的模型来分别解决，包括了传统的机器学习模型（LightGBM和Catboost）和深度学习模型（LSTM和Conv1d），较好地解决了三份数据各有不同的问题；

3. 针对城市A和城市B的分布不一致的情况，设计了两种迁移学习的方案，预训练+微调和嫁接学习，较好地解决了数据分布不一致问题，在三份数据集上均取得了较优的结果。

展望：希望这套解决方案能成功应用到城市C、D、E......中去。
