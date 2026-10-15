# M³RD: Missing-Modality Multimodal Rumor Detection

基于 PyTorch 的中文图文谣言检测研究项目。输入新闻标题和配图，输出真实新闻或虚假新闻分类结果；围绕模态缺失、事件对抗学习和知识蒸馏开展对比实验，并提供 Gradio 演示界面。

本文依据当前目录中的源码与数据整理。项目包含多个实验阶段的脚本，部分分支尚需补齐文件或调整接口，具体见「当前复现限制」。

## 1. 模型与实验设计

| 实验分支 | 文本编码 | 图像编码 | 融合与训练方式 |
| --- | --- | --- | --- |
| 根目录 Baseline | TextCNN | 预训练 ResNet18 | 拼接特征后进行二分类 |
| EANN | TextCNN，输出 32 维 | 预训练 ResNet18，映射为 32 维 | 特征拼接；真假分类器 + 经梯度反转层（GRL）连接的事件鉴别器 |
| EANN + KD | 同 EANN | 同 EANN | 增加单模态教师的 KL 输出蒸馏和 MSE 特征对齐 |
| TGA | 带位置编码的 Transformer Encoder | 预训练 ViT-B/16 | 模态注意力、事件对抗学习，以及仅对完整图文样本启用的相似度门控 |
| TGA + KD | 同 TGA | 同 TGA | Transformer 文本教师与 ViT 图像教师指导学生 |
| 跨事件泛化 | 共用 EANN 的文本和图像编码器 | 共用 EANN 的文本和图像编码器 | 比较普通拼接模型 `plain` 与事件对抗模型 `eann` |

ResNet18 和 ViT 的预训练主干参数被冻结，投影层或替换后的分类头参与训练。TGA 的文本编码器在项目中训练，并非预训练中文语言模型。

蒸馏训练中，文本缺失样本使用图像教师，图像缺失样本使用文本教师。总损失为：

```text
L = L_真假分类 + L_事件分类
    + kd_weight × (L_KL蒸馏 + feature_weight × L_MSE特征对齐)
```

事件分类分支通过 GRL 反转传向特征提取器的梯度。蒸馏项只在对应模态缺失的样本上计算。

## 2. 目录结构

```text
Multimodal/
├── README.md
├── app.py                      # Baseline 的 Gradio 演示，使用 CPU
├── models.py                   # TextCNN + ResNet18 基础检测器
├── dataset_utils.py            # 根目录数据集与图像预处理
├── Training Loop.py            # 基础模型训练入口，文件名含空格
├── modal_missing.py            # 基础模型模态缺失评估
├── word_text.py                # 从训练集构建词表，输出 vocab1.json
├── vocab.json                  # 训练和推理默认使用的词表
├── best_model.pth              # 基础模型权重
├── draw_pic.py                 # 根据脚本内录入的实验数据绘图
├── testqwen.py                 # 独立的外部接口测试，不参与主训练流程
├── CFND_dataset/
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── test_data.csv
│   ├── images/                 # 新闻配图
│   ├── merge.py                # CSV 合并辅助脚本
│   └── README.md               # 数据集说明与引用
├── EANN/
│   ├── models_EANN.py
│   ├── teacher_models.py
│   ├── train_baseline.py
│   ├── train_teachers.py
│   ├── train_distillation.py
│   ├── evaluate_kd_ablation.py
│   └── README_experiment.md
├── TGA/
│   ├── models_TGA.py
│   ├── teacher_models.py
│   ├── dataset_utils.py        # 支持随机模态缺失及缺失标记
│   ├── experiment_utils.py     # 路径、加载器、种子及指标等公共工具
│   ├── train_baseline.py
│   ├── train_teachers.py
│   ├── train_distillation.py
│   └── modal_missing.py
├── event_generalization/
│   ├── split_by_event.py       # 按事件重新划分数据
│   ├── train_compare.py        # plain 与 eann 对比
│   ├── models.py
│   ├── utils.py
│   ├── data/                  # 重新划分后的 CSV
│   └── outputs/               # 权重与评估结果，smoke/ 存放快速检查结果
├── bestmodel/                  # 历史模型权重归档
├── kd/                         # 已弃用的早期蒸馏实验
└── venv/                       # 当前本地虚拟环境
```

`kd/` 已在其说明文件中标记弃用。`bestmodel/` 为历史归档；训练脚本默认向各自分支目录写权重，不会自动写入此目录。

## 3. 环境准备

本地虚拟环境记录的 Python 版本为 **3.10.8**。当前未提供 `requirements.txt` 或 `pyproject.toml`；以下安装命令依据源码导入整理，不代表已经锁定并验证的完整复现环境。

在 Windows PowerShell 中，从项目根目录执行。可使用已有的 `venv`；需要新环境时：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision pandas numpy Pillow jieba scikit-learn gradio matplotlib
```

`testqwen.py` 额外依赖 `requests`，通过 `DASHSCOPE_API_KEY` 环境变量读取接口密钥。使用 GPU 时需安装与本机环境适配的 PyTorch 与 torchvision。

后续示例的 `python` 应指向所选虚拟环境，也可以直接替换为 `.\venv\Scripts\python.exe`。预训练 ResNet18 / ViT 在本地缓存不存在时会尝试下载权重；离线运行前应准备缓存。

Git 仓库不包含虚拟环境、模型权重和 `CFND_dataset/images/` 下的数据图片；本地已有文件不会被删除。克隆后需按 [数据集说明](CFND_dataset/README.md) 准备图片，并自行训练或取得与当前模型和词表匹配的权重，才能执行训练和演示。

## 4. 数据与预处理

### CSV 格式

当前三个数据划分均包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `num` | 样本编号 |
| `title` | 新闻标题，作为文本输入 |
| `image` | 相对于 `CFND_dataset/` 的图片路径 |
| `label` | `0` 为真实新闻，`1` 为虚假新闻 |
| `event` | 事件类别编号；当前数据为 `0`～`9` |

读取图像的路径为 `CFND_dataset / image`。当前数据文件统计如下：

| 划分 | 样本数 | 真实（0） | 虚假（1） |
| --- | ---: | ---: | ---: |
| 训练集 | 15,998 | 10,323 | 5,675 |
| 验证集 | 5,333 | 3,407 | 1,926 |
| 测试集 | 5,334 | 2,664 | 2,670 |
| 合计 | 26,665 | 16,394 | 10,271 |

编写本文时已核对三个 CSV 中引用的图片路径均存在，未逐张检查图片能否解码。数据集来源与论文引用见 [CFND 数据集说明](CFND_dataset/README.md)。

### 预处理规则

- 文本：`jieba.lcut` 分词，默认最多 50 个 token；超出截断，不足填充。
- 词表：根目录 `vocab.json` 当前包含 13,977 个条目，`<PAD>=0`、`<UNK>=1`。
- 图像：转为 RGB，缩放为 `224 × 224`，转 Tensor，并使用 ImageNet 均值和标准差归一化。
- 图片读取失败：数据加载器使用黑色占位图，随后继续执行预处理。
- 人工模态缺失：将文本序列或归一化后的图像 Tensor 置零，与读取失败时的黑色占位图不同。

TGA 训练中的 `--missing-rate` 表示每个样本触发缺失的概率；触发后以各 50% 的概率丢弃文本或图像。`1.0` 表示每个样本都缺失一个模态，不表示两个模态同时缺失。TGA 数据集返回六项：文本、图像、真假标签、事件标签、文本缺失标记、图像缺失标记。

`word_text.py` 从训练集统计词频，保留出现至少两次的词，但输出名为 `vocab1.json`，训练默认读取 `vocab.json`。若更换词表，应统一所有读取路径并重新训练，不能仅替换词表后沿用旧权重。

## 5. 基础模型训练与演示

以下命令均从项目根目录执行：

```powershell
# 训练基础模型，保存为根目录 best_model.pth
python "Training Loop.py"

# 对基础模型进行模态缺失评估
python modal_missing.py

# 加载已有基础模型，启动本地 Gradio 演示
python app.py
```

基础训练脚本固定使用 CPU、10 个 epoch、batch size 32 和学习率 `1e-3`，依据验证集准确率保存最佳权重。更改这些设置需要编辑脚本。

`app.py` 加载根目录的 `vocab.json` 和 `best_model.pth`，要求同时输入文本并上传图片，显示分类结果及 softmax 置信度。启动后打开终端给出的本地地址；代码设置 `share=False`。该界面接入的是基础模型。

## 6. EANN 与知识蒸馏

**当前 EANN 分支缺少 `experiment_utils.py` 和 `dataset_utils.py`，以下为补齐依赖后的预期执行顺序，当前不能直接完整运行。**

```powershell
python EANN/train_baseline.py --epochs 20 --output eann_best_model.pth
python EANN/train_teachers.py --epochs 15
python EANN/train_distillation.py --epochs 20 --missing-rate 1.0
python EANN/evaluate_kd_ablation.py --rates "0,0.2,0.5,0.8,1.0"
```

预期在 `EANN/` 中生成：

- `eann_best_model.pth`：不使用 KD 的学生。
- `Text_Teacher_best.pth` 和 `Image_Teacher_best.pth`：单模态教师。
- `eann_distilled_feature_best.pth`：蒸馏学生。
- `kd_ablation_test_results.csv`：完整图文、丢弃文本及丢弃图像条件下的 Accuracy、Precision、Recall 和 F1。

EANN 教师实际使用 TextCNN 和 ResNet18。评估默认以虚假新闻（`label=1`）为正类，指标以 0～1 的数值写入 CSV；控制台按百分比显示。

已有结果可查阅 [KD 消融准确率表](EANN/kd_ablation_accuracy_table.md) 和分支中的 CSV。这些是历史实验记录，本次文档整理未重新训练或验证其中数值。

## 7. TGA 与知识蒸馏

```powershell
# 不使用知识蒸馏的 TGA
python TGA/train_baseline.py --epochs 20 --missing-rate 0.0

# 当前脚本仅训练图像教师，文本教师训练调用已被注释
python TGA/train_teachers.py --epochs 15

# 需同时存在与当前词表、架构匹配的文本和图像教师权重
python TGA/train_distillation.py --epochs 20 --missing-rate 0.5
```

TGA 的工具模块根据脚本位置解析数据集和根目录词表路径。默认输出位于 `TGA/`：`tga_best_model.pth`、`Text_Teacher_best.pth`、`Image_Teacher_best.pth`、`tga_distilled_feature_best.pth`。工具中的变量虽然仍名为 `EANN_DIR`，实际指向 `TGA/`。

| 参数 | Baseline 默认值 | KD 默认值 | 说明 |
| --- | ---: | ---: | --- |
| `--epochs` | 20 | 20 | 训练轮数 |
| `--batch-size` | 32 | 32 | 批大小 |
| `--lr` | 0.001 | 0.0001 | 学习率 |
| `--num-events` | 50 | 50 | 事件分类头输出数 |
| `--missing-rate` | 0.0 | 0.5 | 训练时的随机模态缺失概率 |
| `--temperature` | — | 5.0 | KL 蒸馏温度 |
| `--kd-weight` | — | 0.5 | 蒸馏总权重 |
| `--feature-weight` | — | 10.0 | MSE 特征对齐权重 |
| `--seed` | 42 | 42 | 随机种子 |

可通过 `--output` 指定学生权重文件名，通过 `--help` 查看参数。验证集默认不模拟模态缺失，学生权重按验证准确率选取。当前已有两种教师权重文件，但从头复现文本教师时需先恢复 `train_teachers.py` 中对应的训练调用。

`TGA/modal_missing.py` 尚有数据批次解包问题，修正前不作为可直接运行的评估入口。

## 8. 跨事件泛化实验

该实验将现有 train / val / test CSV 合并后按 `event` 重新划分，输出到 `event_generalization/data/`，不改写原始 CSV：

| 划分 | 事件编号 |
| --- | --- |
| 训练 | `2, 9, 5, 6` |
| 验证 | `3` |
| 测试 | `0, 1, 4, 7, 8` |

```powershell
python event_generalization/split_by_event.py
python event_generalization/train_compare.py --epochs 20
```

快速检查流程：

```powershell
python event_generalization/train_compare.py --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

默认对比 `plain` 和 `eann`，两者共用同样的编码器和分类头结构，主要区别是是否启用事件对抗训练。此处的 `plain` 与根目录 Baseline 的特征维度不同。

完整实验向 `event_generalization/outputs/` 写入两个模型的最佳权重和 `event_generalization_results.csv`；限制批次数的快速检查写入 `outputs/smoke/`。数据划分脚本另生成 `data/split_summary.csv`。

该实验按现有事件编号隔离数据；事件编号的生成过程和词表来源也会影响泛化结论，需要与划分方案一起记录。

## 9. 绘图与结果管理

```powershell
python draw_pic.py
```

绘图脚本使用内部录入的数值，生成 Baseline / EANN / TGA 及其蒸馏版本的对比图，不会自动读取最新评估 CSV。更新实验后需同步修改绘图数据。中文字体配置为 `SimHei`，运行环境需提供该字体。

训练可能覆盖同名权重，建议用 `--output` 区分实验，并保存词表、事件数、随机种子、缺失率和评估结果。当前权重主要保存为 `state_dict`，不会自动附带这些实验配置。

## 10. 当前复现限制

1. **EANN 文件缺失**：`EANN/train_baseline.py --help` 已实际检查，因找不到 `experiment_utils` 报错。该目录也缺少独立的数据集模块；根目录数据集不支持其预期的 `missing_rate` 参数及事件、缺失标记返回接口。`__pycache__` 中的缓存不能替代完整源文件。应恢复符合这些调用约定的源码。
2. **TGA 教师训练不完整**：当前入口只调用图像教师训练。从头复现需恢复文本教师调用，再执行蒸馏；两个教师权重不能与 EANN 分支混用。
3. **TGA 评估接口不一致**：`TGA/modal_missing.py` 按四项解包批次，而其数据集返回六项；还使用相对当前工作目录的路径。修正解包和路径后再评估。
4. **事件数与权重必须匹配**：当前数据有 10 个事件，但 EANN / TGA 学生脚本默认输出 50 类。新训练可按设计调整；加载现有权重时事件分类头尺寸必须保持一致。
5. **历史预处理脚本需审查**：`EANN/clustering.py` 当前仅对验证和测试 CSV 联合聚类并覆盖 `event` 列，不能直接作为三份数据一致的事件生成流程。`CFND_dataset/merge.py` 的预期列名还把 `image` 写为 `imag`。现有 CSV 已准备好，常规训练无需运行这两个脚本。
6. **环境与结果尚未完整复验**：目前没有统一依赖锁定文件或自动化测试套件。本文核对了源码、数据统计、图片路径及部分命令行入口，未执行完整训练、全量推理或 Gradio 交互验证。

分支说明可继续参考 [EANN 实验说明](EANN/README_experiment.md)、[跨事件实验说明](event_generalization/README.md) 和 [旧 KD 分支说明](kd/README_deprecated.md)；运行时应结合本文所列的当前源码限制。
