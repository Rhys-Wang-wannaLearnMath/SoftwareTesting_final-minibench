## 执行测试
(1)qwen_cuda是candle项目编译得来，存放在对应文件夹：
```bash
python3 autoTest.py --name qwen_cuda --model 0.5b
python3 autoTest.py --name qwen_cuda --model 1.8b
python3 autoTest.py --name qwen_cuda --model 2-0.5b
python3 autoTest.py --name qwen_cuda --model 2-1.5b

（2）进行评测：
```bash
python3 autoEvaluate.py --target qwen_cuda_0.5b qwen_cuda_2-0.5b qwen_cuda_1.8b qwen_cuda_2-1.5b

（3）将评分结果进行总结对比：
```bash
python3 autoSum.py --target 1
