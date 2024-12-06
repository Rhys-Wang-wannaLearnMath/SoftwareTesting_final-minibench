#!/usr/bin/env python3
# python3 autoTest.py --name *** --model 可选

import os
import subprocess
import argparse
import sys


def run_test(name, model, script_name, question_file):
    """执行单个测试"""
    command = ["python3", script_name, "--name", name, "--questions", question_file]

    # 如果提供了model参数，则添加
    if model:
        command.extend(["--model", model])

    print(f"\n执行命令: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="自动执行所有测试")
    parser.add_argument("--name", type=str, required=True, help="可执行文件名")
    parser.add_argument("--model", type=str, required=False, help="模型编号")

    args = parser.parse_args()

    # 所有测试配置
    tests = {
        "单选题自动测试.py": [
            "单选题_事实性回答_问题集.json",
            "单选题_基础阅读理解_问题集.json",
            "单选题_情感分析_问题集.json",
        ],
        "多选题自动测试.py": [
            "多选题_事实性回答_问题集.json",
            "多选题_基础阅读理解_问题集.json",
            "多选题_情感分析_问题集.json",
        ],
        "问答题自动测试.py": [
            "问答题_事实性回答_问题集.json",
            "问答题_基础阅读理解_问题集.json",
            "问答题_情感分析_问题集.json",
        ],
    }

    # 记录执行结果
    results = []

    # 执行所有测试
    for script, question_files in tests.items():
        print(f"\n开始执行 {script} 的测试...")

        # 检查脚本文件是否存在
        if not os.path.exists(script):
            print(f"警告: {script} 不存在，跳过相关测试")
            continue

        for question_file in question_files:
            if not os.path.exists(question_file):
                print(f"警告: {question_file} 不存在，跳过此测试")
                continue

            print(f"\n正在测试: {question_file}")
            success = run_test(args.name, args.model, script, question_file)
            results.append(
                {"script": script, "question_file": question_file, "success": success}
            )

    # 打印执行摘要
    print("\n执行摘要:")
    print("=" * 50)
    success_count = sum(1 for r in results if r["success"])
    print(f"总测试数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")

    if len(results) - success_count > 0:
        print("\n失败的测试:")
        for r in results:
            if not r["success"]:
                print(f"- {r['script']} : {r['question_file']}")


if __name__ == "__main__":
    main()
