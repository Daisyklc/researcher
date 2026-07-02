# 需求表格

> 也可直接提供此文件的截图（`.png` / `.jpg`），将由 DeepSeek 多模态视觉模型自动识别。

| 字段 | 内容 |
|------|------|
| 项目名称 | Titanic 生存预测 Baseline |
| 任务类型 | 二分类 (binary classification) |
| 目标 | 根据乘客特征预测 Survived (0/1) |
| 评估指标 | Accuracy |
| 数据文件 | train.csv, test.csv |
| 推荐框架 | sklearn |
| 推荐模型 | GradientBoostingClassifier |
| 交叉验证 | StratifiedKFold, 5 折 |
| 特征工程 | 从 Name 提取 Title; FamilySize; Age/Fare 填补与分箱; one-hot 编码 |
| 产出文件 | submission.csv (PassengerId, Survived) |
| 项目来源 | 优先参考 GitHub 上 titanic sklearn baseline 开源实现 |
| 约束 | 控制内存占用; 脚本需自包含可运行 |

## 补充说明

- 需要阅读项目依赖文件 (requirements.txt 等) 并列出开源软件名称
- 需要检索网上关于 Titanic sklearn baseline 的近期讨论
- 输出应包含已阅读的文件路径与项目业务逻辑说明
