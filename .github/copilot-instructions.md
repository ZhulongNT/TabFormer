# Copilot Instructions for TabFormer

## 项目架构与主要组件
- 本项目实现了 Tabular Transformers（TabFormer），用于多变量时间序列的建模。
- 主要模块分布在 `models/`（Transformer 结构、BERT/GPT2 变体）、`dataset/`（数据集处理与特定数据集支持）、`misc/`（工具函数）、`main.py`（训练/推理入口）。
- 数据集位于 `data/credit_card/`（合成信用卡交易数据，24M条记录，12字段）和 `data/card/`（PRSA 数据集，需手动下载）。

## 关键开发流程
- 环境建议：使用 `conda env create -f setup.yml` 安装依赖，推荐 Python 3.7。
- 训练入口：
  - Tabular BERT：
    ```
    python main.py --do_train --mlm --field_ce --lm_type bert --field_hs 64 --data_type [prsa/card] --output_dir [output_dir]
    ```
  - Tabular GPT2：
    ```
    python main.py --do_train --lm_type gpt2 --field_ce --flatten --data_type card --data_root [path_to_data] --user_ids [user-id] --output_dir [output_dir]
    ```
- 主要参数说明见 `args.py`，如 `--data_type`、`--mlm`、`--field_hs`、`--lm_type`、`--user_ids`。

## 项目约定与模式
- 数据预处理与编码在 `data/credit_card/preprocessed/`，如 `.encoded.csv`、`.pkl` 文件。
- 训练日志与输出在 `output/` 及其子目录。
- 采用 HuggingFace Transformers 作为底层依赖，部分模块（如 DataCollator）有自定义实现。
- 代码风格以模块化为主，模型、数据、工具分离，便于扩展。

## 集成与外部依赖
- 需安装 `git-lfs` 获取大数据文件，或使用 Box 链接手动下载。
- 依赖库版本见 `requirements.txt` 和 `setup.yml`，如 Pytorch、scikit-learn、pandas。

## 重要文件/目录参考
- `main.py`：训练/推理主入口。
- `models/`：Transformer 结构与自定义损失。
- `dataset/`：数据集加载与处理。
- `misc/utils.py`：常用工具函数。
- `data/credit_card/README.md`：数据说明与获取方式。
- `requirements.txt`、`setup.yml`：依赖管理。

## 示例：自定义训练参数
- 可在 `main.py` 通过命令行参数灵活配置模型结构与数据来源。
- 例如：
  ```
  python main.py --do_train --lm_type bert --data_type card --field_hs 128 --output_dir output/exp1
  ```

---
如需补充特殊约定、调试技巧或集成细节，请反馈。