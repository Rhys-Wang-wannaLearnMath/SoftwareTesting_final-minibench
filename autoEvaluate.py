# python3 autoEvaluate.py --target qwen_cuda_2-1.5b qwen_cuda （必须是多个）

import os
import argparse
import subprocess
import sys


def validate_target(target: str) -> bool:
    """验证target参数格式是否正确"""
    parts = target.split("_")
    if len(parts) not in [2, 3]:  # name_cuda 或 name_cuda_model
        return False
    if parts[1] != "cuda":
        return False
    return True


def get_matching_files(target: str, question_type: str, problem_type: str) -> str:
    """获取匹配的文件名"""
    pattern = f"{target}_{question_type}_{problem_type}_问题集.json"
    return pattern


def run_evaluation(targets: list):
    """执行评分脚本"""
    # 评估类型配置
    eval_configs = [
        {
            "problem_type": "事实性回答",
            "question_types": ["单选题", "多选题", "问答题"],
        },
        {
            "problem_type": "基础阅读理解",
            "question_types": ["单选题", "多选题", "问答题"],
        },
        {"problem_type": "情感分析", "question_types": ["单选题", "多选题", "问答题"]},
    ]

    # 验证target格式
    invalid_targets = []
    for target in targets:
        if not validate_target(target):
            invalid_targets.append(target)

    if invalid_targets:
        print(f"错误: 以下target参数格式不正确: {invalid_targets}")
        print("正确格式: name_cuda 或 name_cuda_model")
        sys.exit(1)

    # 对每种评估类型执行评分
    for config in eval_configs:
        problem_type = config["problem_type"]
        print(f"\n开始评估 {problem_type}...")

        for question_type in config["question_types"]:
            # 收集所有目标文件
            files_to_evaluate = []
            missing_files = []

            for target in targets:
                filename = get_matching_files(target, question_type, problem_type)
                if os.path.exists(filename):
                    files_to_evaluate.append(filename)
                else:
                    missing_files.append(filename)

            # 报告缺失文件
            if missing_files:
                print(f"\n警告: 未找到以下文件:")
                for f in missing_files:
                    print(f"- {f}")

            # 如果有匹配的文件则执行评分
            if files_to_evaluate:
                script_name = f"{question_type}评分.py"
                command = ["python3", script_name, "--jsons"] + files_to_evaluate

                print(f"\n执行命令: {' '.join(command)}")
                try:
                    subprocess.run(command, check=True)
                    print(f"成功完成 {question_type} 评分")
                except subprocess.CalledProcessError as e:
                    print(f"警告: {question_type} 评分执行失败: {e}")
                except Exception as e:
                    print(f"警告: 执行 {question_type} 评分时发生错误: {e}")
            else:
                print(f"\n跳过 {question_type} 评分: 没有找到匹配的文件")

    print("\n所有评估任务完成")


def main():
    parser = argparse.ArgumentParser(description="自动执行模型评分")
    parser.add_argument(
        "--target",
        nargs="+",
        required=True,
        help="评估目标, 格式: name_cuda 或 name_cuda_model",
    )

    args = parser.parse_args()

    if len(args.target) < 2:
        print("错误: 至少需要提供2个target参数")
        sys.exit(1)

    run_evaluation(args.target)


if __name__ == "__main__":
    main()
