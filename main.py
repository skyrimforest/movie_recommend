'''
@Project ：movie_recommend 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/18 21:44 
'''

target = "1、二分类评估指标有哪些？ 2、AUC是什么？怎么画的，怎么计算的？ 3、BPR Loss是什么？ 4、NDCG指标是什么？怎么计算的？ 5、SVD原理？ 6、LightGBM和XGBoost的原理和区别？ 7、逻辑回归原理？ 8、DeepFM模型原理？ 9、Wide&Deep模型原理？ 10、准确率、召回率，精准率定义和区别？ 11、推荐业务流程介绍？ 12、深度学习中防止过拟合的方法？ 13、Dropout在预测和训练阶段的区别？ 14、RNN原理？ 15、RNN有哪些变种？原理介绍？ 16、Word2Vec的原理介绍？ 17、Word2Vec中的CBow和Skip-gram是？ 18、LGB和XGB对缺失值的处理方式区别？ 19、RF和LGB在方差和偏差的区别？ 20、Transformer介绍？ 21、Self-attention和Target-attention区别？ 22、L1和L2的区别？ 23、Batch-norm和Layer-norm介绍和区别？ 24、Batch-norm使用时需要注意什么？ 25、激活函数介绍和区别？ 26、梯度爆炸或者为0时，如何解决？ 27、GAUC是什么？"
res = target.split("、")
idx=0
for i in res:
    i = i.lstrip('0123456789 ').rstrip('0123456789 ')
    print(f"{idx} {i}")
    idx+=1