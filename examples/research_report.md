# 文献与策略研究报告：Titanic 生存预测

## 1. 问题概述
Titanic 是经典的二分类入门竞赛，目标是根据乘客的属性（舱位、性别、年龄、票价、登船港口等）
预测其是否在海难中幸存（`Survived` ∈ {0, 1}）。评估指标为 **Accuracy**。

## 2. 关键结论（Baseline 应落地的要点）
- **性别是最强单特征**：女性幸存率显著高于男性，任何模型都应保留 `Sex`。
- **舱位等级 `Pclass`** 与幸存率强相关（1 > 2 > 3）。
- **称谓 `Title`（Mr/Mrs/Miss/Master/Rare）** 从 `Name` 提取后，能同时编码性别、年龄段与社会地位，
  是公认的强特征，且能辅助填补 `Age` 缺失。
- **家庭规模 `FamilySize` 与 `IsAlone`**：中等家庭规模幸存率更高，单独出行者偏低。
- **`Fare`/`Age` 分箱**：树模型对分箱不敏感，但分箱能提升线性/近邻类模型稳健性。

## 3. 缺失值处理建议
| 字段 | 缺失情况 | 处理策略 |
|------|----------|----------|
| Age | ~20% 缺失 | 按 `Title`(或 Pclass+Sex) 分组中位数填补 |
| Embarked | 极少缺失 | 用众数 `S` 填补 |
| Fare | 测试集个别缺失 | 按 Pclass 中位数填补 |
| Cabin | 大量缺失 | 取首字母作为 `Deck`，缺失记为 `Unknown` |

## 4. 模型选型建议
- **首选 Baseline**：`GradientBoostingClassifier` / 随机森林，无需调参即有 ~0.78-0.80 accuracy。
- **交叉验证**：使用 `StratifiedKFold(n_splits=5)`，报告 OOF 平均准确率，避免过拟合单一划分。
- **进阶（非本次 baseline 范围）**：XGBoost/LightGBM + 特征交叉，或简单 stacking。

## 5. 对生成代码的硬性要求
1. 数据从工作区 `data/` 目录读取（`train.csv` / `test.csv`）。
2. 完成上述特征工程与缺失值填补。
3. 用 5 折分层交叉验证输出 OOF accuracy，并在全量训练后对测试集预测。
4. 生成 `submission.csv`（列：`PassengerId`, `Survived`），保存到工作区根目录。
5. 清晰 `print` 每折得分与最终 CV 均值。
