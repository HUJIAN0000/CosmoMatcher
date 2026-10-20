# 验证记录

验证日期：2026-10-14。主程序 v1.0.0，随附匹配后端 v1.4.3。

## 已完成的检查

- 11 项数值／统计测试通过。
- 使用你提供的 1701 行 Pantheon+ 表、完整 STAT+SYS 协方差和 161 个透镜表完成端到端运行。
- 全局规范化匹配得到 112 个系统、224 个不同物理 SN 标识。
- 使用固定随机种子 20261014 完成 10000 次配对模拟。
- 基线汇总偏差和两类区间覆盖率与独立解析参考在 Monte Carlo 不确定度范围内一致。
- 核查了汇总图。图表为 Matplotlib 科学绘图。
- 在基线之后完成 stress_config.json 的九组统计运行，每组 10000 次配对模拟；真实背景 Omega_m=0.28/0.30/0.32，容差为 3%/5%/7%。
- 另以 100 次小规模运行检查过外部匹配表输入及 w=-0.9 分支；w=-0.9 不属于此次九组论文实验。
- 原始后端 46 项回归测试通过，并再次复现了 112 个目标的完整匹配表和所有导出协方差矩阵。

## 输入及可用样本

输入协方差首行实际为 1701。最大原始非对称为约 3×10^-8，按记录进行对称化；完整母矩阵的最小特征值约为 0.00080278 mag²。没有另加测量误差对角项或额外固有散度。

单点匹配、常数 GLS、线性 GLS 对完整透镜输入分别可用 112、143、128 个系统。三者共同样本为 111 个系统。单点样本中索引 160 的系统（zl=0.527、zs=2.2649）在源锚点的候选数量不足以识别线性斜率，因此没有强行进入线性 GLS。

这仅是可用性统计，不应拿三个方法各自不同的目标集合直接比较 RMSE 或覆盖率。

## 10000 次基线结果

真值和匹配背景均采用 Omega_m=0.3，容差为 5%。下表使用三方法共同的 111 个系统，覆盖率采用 log-q 变换区间。

| 方法 | 有符号平均偏差 | pooled RMSE | 68% 覆盖率 | 95% 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 单点匹配 | -0.003537 | 0.053189 | 67.7432% | 94.8881% |
| 常数 GLS | -0.002152 | 0.015686 | 57.6010% | 88.2974% |
| 线性 GLS | -0.000380 | 0.017995 | 68.0005% | 95.0203% |

完整精度、MCSE、解析参考、delta 区间覆盖率和逐目标结果均在 sample_results 中。主要的单点／常数 GLS 对照有 112 个共同系统，也单独输出于 metrics_summary.csv。

该结果说明在此规定的模拟中，较小 RMSE 不一定伴随正确的区间覆盖率；不能把结果推广为所有数据或所有下游似然的结论。基线没有模拟红移误差、透镜模型误差、光度演化或非平直几何。

## 复现

运行 benchmark_config.json 将在 results 目录重新生成 10000 次基线结果。sample_results 是已经运行的参考输出，run_manifest.json 保存原始文件 SHA256、实际配置和库版本。

首次 1000 次运行（含全局规范化匹配）用时约 26.8 秒，其中规范化约 22.5 秒。随后利用验证过的匹配缓存运行 10000 次基线约 2.8 秒。这是本次环境的测量，不是对你的 Windows 设备的耗时保证。

stress_config.json 的全部九组已运行，结果在 stress_results 中。均值偏差与两类区间的 68%/95% 覆盖率，在所有汇总行上与解析参考的最大差异均不超过 2.20 个 MCSE。3% 线性 GLS 的重尾例外需要单独解释：共同目标上的模拟 RMSE 为 1.473569，解析 RMSE 为 1.902746，经验 MCSE 约 0.386。有限次模拟不能精确估计该重尾二阶矩，不能仅用覆盖率接近标称值评价它的实用精度。

3% 源窗口的一个共同目标为 z_s=1.200，两颗候选在 1.23225/1.23597。线性权重为 9.669355/-8.669355，使 sigma_logq 达 2.081065。这个外推目标没有被事后删除，没有新增自适应方差阈值。

真实观测结果保存在 observed_results。使用实际 m_b_corr 重现了原有常数 GLS 数值，并增加线性 GLS 及共同目标的配对协方差。真实数据没有未知真值的 bias/coverage 指标。

## 与论文相对应的运行命令

```bash
python cosmomatcher_benchmark.py --config benchmark_config.json
python cosmomatcher_benchmark.py --config stress_config.json --output stress_rerun
python reproduce_observed_gls.py
python make_paper_figures.py --results stress_rerun --output paper_figures
python -m unittest discover -s tests -v
python -m unittest discover -s backend/tests -v
```

stress_results/run_manifest.json 和 observed_results/observed_manifest.json 保存实际输入、配置和来源；stress_results/analytic_checks.json 保存解析核对结果。
