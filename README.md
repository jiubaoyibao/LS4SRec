# LS4SRec

This is the PyTorch implementation of LS4SRec described in my paper:

> Self-supervised Graph Neural Sequential Recommendation with Disentangling Long
and Short-Term Interests .

# Overview
The sequential recommendation systems capture users' dynamic behavior patterns to predict their next interaction behaviors. 
The existing sequence recommendation approches do not fully realize the importance of longshort term interest disentangling, and it is difficult to deal with the negative impact caused by
data sparsity and noise information. This thesis proposes a self-supervised graph neural network sequence recommendation model (LS4SRec) based on global context information and
long-short term interest disentanglement. Specifically, the model first constructs a global item
transition graph to provide global context information for interaction sequences, and enhances
sequence representation through subgraph sampling and graph neural network. Then two separate 
encoders are designed based on the self attention mechanism and the gated recurrent unit
to encode the user’s long and short-term interests respectively. At the same time, interest evolution sequence is constructed by using interest allocation matrix and sequence pooling, and
user interest evolution information is extracted based on GRU. Finally, user’s long-short term
interests and interest evolution information are aggregated for future interactions prediction
adaptively. In addition, two auxiliary learning objectives have been proposed to ensure the
consistency between different augmented representations induced by same sequence and the
real disentanglement of long-short term interests. Experimental results demonstrate that the
LS4Sec can improve the user’s long-term and short-term interest representation by disentangling interests, and outperform the state-of-the-art (SOTA) sequence recommendation models.

Figure 2 shows the overall framework of LS4SRec. Observe that LS4SRec has the following main components: 1) Sequence Enhancement, 2) Long-Short Term Interest Model, 3) Interest Evolution Modeling, and 4) Interaction Prediction.
![avatar](figures/LS4SRec.png)

# Requirement:
This implementation is based on pytorch geometric. To run the code, you will need the following dependencies:

- python 3
- torch 
- torch-geometric 
- tqdm 
- pickle 
- scipy 

# Datasets:

## data format
Taking home dataset as an example
```shell script
    home.txt 
    one user per line
    user_1 item_1,item_2,...
    user_2 item_1,item_2,...
    0 1,2,3,4,5,6,7,8
    1 5,9,10,11,12
    ...
    
    all_train_seq.txt
    have the same format as home.txt, but remove the last and the second last interaction item
    0 1,2,3,4,5,6
    1 5,9,10
    ...
    
    train.pkl
    have four list, containing user_id, item_seq, target_item, seqence_len
    (
    [0, 0, 0, 0, 0, 1, ...], 
    [[1, 2, 3, 4, 5],
     [1, 2, 3, 4],
     [1, 2, 3],
     [1, 2],
     [1],
     [5, 9]
     ...],
    [6, 5, 4, 3, 2, 10, ...],
    [5, 4, 3, 2, 1, 2, ...]
    )
    
    test.pkl and valud.pkl
    have the same format as train.txt
```

## build Global Item Transition Graph
Using all observed data(all_train_seq.txt) to build weighted item transition graph, execute:
```shell script
    python build_witg.py 
```

Figure 1 shows an example about the transition graph without edge weight normalization.
![avatar](figures/GITG.jpg)


# Usage:
For example, to run LS4SRec under Home dataset, execute:
```shell script
    python runner.py --data_name='home'
```

You can also change parameters according to the usage, which is also including detailed explanation of each hyper-parameter:
```shell script
    python runner.py -h
```

# Pseudo
Algorithm 4.1 show the pseudo code of LS4SRec.![avatar](figures/Pseudo.png)

# Code
```shell script
    | > datasets      实验数据集 与 全局Item转移图构建代码,构建逻辑详见论文4.3.1节(序列增强);
    | > dataset.py    实现自定义Torch中Dataloader的数据读取和划分依据;
    | > model.py      实现Baseline模型 GCL4SR,LS4SR,S^3-Rec, CL4SRec等;
    | > Newmodel.py   实现本文模型LS4SRec, 及其变体模型;
    | > runner.py     项目运行的主函数,定义模型超参,函数接口等;调用model/Newmodel/trainner/utils完成模型整体实验流程;
    | > trainner.py   实现模型开展实验的训练,验证,测试逻辑;
    | > utils.py      实现模型评估指标计算,Earrly Stop,点击矩阵计算;
    | > README.md     项目说明
```

# Contact
This implementation is partly based on [S3-Rec](https://github.com/aHuiWang/CIKM2020-S3Rec)  and [GCL4SR](https://github.com/sdu-zyx/GCL4SR/blob/main/runner.py) modules.
If you have any questions or concerns, please send an email to hlsun@xjtu.edu.cn.
