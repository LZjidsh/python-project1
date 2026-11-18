# Project 1 — 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

深圳大学《Python程序设计》Project 1 提交。在不使用 PyTorch / OpenCLIP / GPU 的前提下，
仅用 NumPy 实现 OpenCLIP ViT-B/32 图像编码器的前向计算，输出与课程提供的参考输出在
给定容差内数值一致。**只修改了 `image_encoder.py`**，其余文件均按课程原始版本使用。

- 评测文件：`image_encoder.py`（六个 TODO 组全部完成）
- 实验报告：Overleaf 项目（正文 4 页）

---

## 1. 环境要求

Python 3.9 及以上，仅需 NumPy 与 Pillow（无 GPU、无 PyTorch、无需联网）。
`imageio` 只被课堂示例 02 使用。

```bash
python -m pip install -r requirements.txt
```

已验证环境：Python 3.13.14 / NumPy 2.5.5 / Pillow 13.0.0。

---

## 2. 如何复现结果

所有命令都在**包含 `image_encoder.py` 的目录**下执行。

```bash
# (1) 校验课程提供的参数、图像与参考数组是否完整
python check.py --stage assets

# (2) 逐阶段与参考输出比对（推荐按实现顺序）
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# (3) 一次跑完全部 8 个阶段
python check.py --stage all
```

全部通过时每阶段打印一行，包含形状、最大/平均绝对误差与容差。例如：

```
patches      PASS shape=(2, 49, 3072) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
output       PASS shape=(10, 512) max_abs=5.439e-07 mean_abs=8.311e-08 (atol=2e-05, rtol=0.0002)
```

---

## 3. 生成输出 token

```bash
# 课程提供的十张图像，按清单顺序
python run.py --output output_tokens.npz

# 也可以编码自己的图像，ID 取文件名主干，顺序与输入一致
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

输出归档包含 `ids` 与形状 `[B,512]` 的 `output_tokens`（float32，每行单位长度）。
批大小自由：B=1、B=3、B=10 的同一行结果相差约 1e-7，重复图像得到逐位相同的行。

---

## 4. 各阶段数值误差

| 阶段 | 形状 | max abs | mean abs | atol | rtol |
| --- | --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | 2e-06 | 2e-04 |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | 2e-04 | 2e-04 |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | 2e-04 | 2e-04 |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | 2e-04 | 2e-04 |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | 2e-04 | 2e-04 |
| output | (10,512) | 5.439e-07 | 8.311e-08 | 2e-05 | 2e-04 |
| batch | (3,512) | 4.619e-07 | 7.632e-08 | 2e-05 | 2e-04 |

残余误差来自 float32 累加次序差异：只有不含归约的 preprocess 与 patches 能完全精确。
额外验证：单张与批内计算相差 5.2e-08，同图两份拷贝输出逐位相同；每行平方长度为
1.000000；`run.py` 可编码不在参考集中的图像（实现中无任何缓存）。

---

## 5. 实现说明（六个 TODO）

| TODO | 方法 | 操作 |
| --- | --- | --- |
| 01 | `ImageEncoder.patchify` | 5 维 reshape + 转置，`[B,3,224,224] -> [B,49,3072]` |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,768] -> [B,12,N,64]` |
| 03 | `MultiHeadAttention.__call__` | 缩放打分、系数加权 v、头合并、输出投影 |
| 04 | `TransformerBlock.__call__` | 两次残差相加 + GELU MLP |
| 05 | `ImageEncoder.embed` | 无偏置 patch 投影、前置 CLS、位置向量 |
| 06 | `ImageEncoder.__call__` | 取 CLS、post-Norm、`[768,512]` 投影、单位化 |

关键约定：

1. patch 为 32×32，224/32 = 7×7 = 49 个 patch，每个按 C,H,W 展平为 3072 维；
2. patch 投影**无偏置**，使用 `patch_weight.T`；
3. 注意力与 MLP 线性层按 **[out,in]** 存储，需乘 `.T`；
4. `visual.proj` 已是 **[in,out] = [768,512]**，**不能转置**；
5. 使用课程给定的精确 GELU 与 Norm epsilon（1e-5），全程无参数更新；
6. 每个 Block 两次残差相加，无 LayerScale。

---

## 6. 踩坑记录（供复现参考）

- **patchify 顺序**：直接 `reshape(B,49,3072)` 形状正确但内容错误（max_abs 3.812）。
  必须先把两个空间轴拆成“patch 序号 + patch 内坐标”再转置，否则每个 patch 会混入
  不同通道的行。**形状检查无法验证顺序。**
- **k 的转置**：k 是 4 维，必须用 `swapaxes(-1,-2)` 只交换最后两轴；`k.T` 会把四个轴
  全部反转（`[B,12,50,64] -> [64,50,12,2]`）。
- **dtype 提升**：`np.sqrt(head_dim)` 返回 float64 标量，用它作除数会让整条前向通路
  变成 float64 并被检查脚本拒绝；转成 `np.float32` 即可。

---

## 7. 文件说明

```
image_encoder.py            评测实现（唯一被修改的文件）
helpers.py                  课程提供：LayerNorm / GELU / 注意力系数 / 单位化
preprocess_image.py         课程提供：缩放、中心裁剪、归一化
runtime.py, run.py          课程提供：权重加载与命令行入口
check.py                    课程提供：分阶段数值检查
config.json                 模型架构与权重布局
requirements.txt            Python 依赖
README.md                   本文件
```

> 注意：`weights/image_encoder.npz`（311 MB）与 `assets/reference_outputs.npz`（7.3 MB）
> 体积较大，按作业要求**不在此仓库重复提交**，请使用课程提供的原始副本。

---

## 8. 说明

`image_encoder.py` 中六个 TODO 组由本人实现；编写过程中使用了 AI 编程助手协助起草
部分 NumPy 表达式，每条表达式均通过运行 `check.py` 的全部阶段及上文额外测试做了
独立验证，最终代码由本人复核提交。
