# SoftwareTesting_final-minibench
This is my final project for Software Testing, a sse student from sysu.

执行步骤：
（1）python3 autoTest.py --name x --model(option) y
输出的文件命名规则：x_y(option)_问题集名称.json
（2）对于其中的单选题和多选题的回答内容，需要人工或者LLM辅助处理，使其符合评分格式：单选题为一个字母；多选题为多个字母连在一起。
（3）python3 autoEvaluate.py --target x_y(option)（必须是多个，单个无法执行，因为无法产生速度基准）
输出的文件命名规则：问题集名称_评分n.json（n代表是此时第几份评分文件）
（4）python3 autoSum.py --target n (进行总结的评分文件编号，只有一个参数)。将对其中的评分情况进行总结。
