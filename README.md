### 摘要
首先，基于PyTorch
框架成功复现了经典的DDPM模型；其次，在网络结构中引入了自注意力机制以增强全局建模能力；第三，采用了不同的噪声调度策略；第四，分析了不同扩散步数和学习率对生成质量的影响；最后，创新性地设计了新的预测对象和采样方法。

### 实验环境
操作系统：ubuntu22.04
<br>深度学习框架：PyTorch 2.1.2
<br>CUDA Version ：12.1
<br>编程语言：Python 3.10.8
<br>CPU：16 vCPU Intel(R) Xeon(R) Platinum 8481C
<br>GPU: NVIDIA RTX 4090 (24GB)

### 数据集
本实验使用的数据集是MNIST，若使用其他数据集自行下载至当前文件夹中，然后修改main.py

### 训练
step1：安装所需要的依赖---- pip install -r requirements.txt
<br>step2：在main.py中设定所需的参数例如lr，batchsize等，确定是否启用attention，余弦噪声调度，以及是否采用新的预测对象和采样方式
<br>step3: 修改权重和图片的保存路径
<br>step4: 运行 python -u "main.py"

### 测试
加载训练好的权重：运行 python -u "test.py"

### 实验结果
![DDPM Result 1](https://raw.githubusercontent.com/frederickkkkk/DDPM/main/result_1.png )<br>
![DDPM Result 2](https://raw.githubusercontent.com/frederickkkkk/DDPM/main/result_2.png )
