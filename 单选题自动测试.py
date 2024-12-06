# 用法：python3 单选题自动测试.py  --name *** (--model ***) --questions ***.json


import os
import subprocess
import argparse
import json
from datetime import datetime
import re


def load_questions(json_path):
    """从JSON文件加载问题"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["test_samples"]
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return None


def create_prompt(question_data):
    """创建简单的问题prompt"""
    options = "\n".join(
        [f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_data["options"])]
    )
    prompt = f"{question_data['question']}\n{options}"
    return prompt


def save_results(results, executable_name, model_number, questions_file):
    """保存测试结果到JSON文件"""
    # 构建输出文件名
    questions_basename = os.path.splitext(os.path.basename(questions_file))[0]
    model_part = f"_{model_number}" if model_number else ""
    output_filename = f"{executable_name}{model_part}_{questions_basename}.json"

    # 构建结果数据
    output_data = {
        "test_info": {
            "executable": executable_name,
            "model": model_number,
            "questions_file": questions_file,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    # 保存到文件
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_filename}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def extract_token_stats(stdout):
    """提取token统计信息"""
    # 匹配类似 "7 tokens generated (81.55 token/s)" 的模式
    pattern = r"(\d+)\s+tokens\s+generated\s+\(([\d.]+)\s+token/s\)"
    match = re.search(pattern, stdout)
    if match:
        return {
            "output_tokens_num": int(match.group(1)),
            "output_tokens_speed": float(match.group(2)),
        }
    return None


def clean_model_output(stdout):
    """清理模型输出，移除调试信息"""
    # 分割输出行
    lines = stdout.split("\n")

    # 需要过滤的调试信息模式
    debug_patterns = [
        r"avx:.+",
        r"temp:.+",
        r"retrieved the files.+",
        r"loaded the model.+",
        r"\d+ tokens generated.+",  # 移除token统计信息，因为我们会单独存储
    ]

    # 过滤非调试信息行
    cleaned_lines = []
    for line in lines:
        if not any(re.match(pattern, line.strip()) for pattern in debug_patterns):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def execute_command(executable_name, directory, prompt, model_number=None):
    """执行命令并返回输出"""
    executable_path = os.path.join(directory, executable_name)

    if not os.path.isfile(executable_path) or not os.access(executable_path, os.X_OK):
        print(f"错误: {executable_path} 不是一个可执行文件或不存在。")
        return None

    command = [f"{executable_path}"]
    if model_number:
        command.extend(["--model", model_number])
    command.extend(["--prompt", f'"{prompt}"'])

    try:
        print(f"正在执行命令: {' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        # 提取token统计信息
        token_stats = extract_token_stats(stdout)

        return {
            "stdout": clean_model_output(stdout),
            "stderr": stderr,
            "return_code": process.returncode,
            "token_stats": token_stats,
        }
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="执行可执行文件并传入问题")
    parser.add_argument("--name", type=str, required=True, help="可执行文件名")
    parser.add_argument("--model", type=str, required=False, help="模型编号")
    parser.add_argument(
        "--questions",
        type=str,
        default="单选题数据集.json",
        help="问题数据集JSON文件路径",
    )

    args = parser.parse_args()
    directory = "/home/wzr/native_CandleExamples"

    questions = load_questions(args.questions)
    if not questions:
        return

    all_results = []

    for question_data in questions:
        prompt = create_prompt(question_data)
        output = execute_command(args.name, directory, prompt, args.model)

        if output:
            # 清理model输出，移除问题重复和选项列表
            raw_output = output["stdout"]
            # 匹配引号内的问题和选项列表
            quote_pattern = rf'\\?".*?(?:\\n[A-D]\. .+?)*\\?"'
            cleaned_output = re.sub(quote_pattern, "", raw_output, flags=re.DOTALL)

            result = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "options": question_data["options"],
                "correct_answer": question_data["correct_answers"][0],
                "execution_time": datetime.now().isoformat(),
                "model_output": cleaned_output.strip(),  # 使用清理后的输出
                "errors": output["stderr"],
                "status": "success" if output["return_code"] == 0 else "failed",
            }

            # 添加token统计信息
            if output["token_stats"]:
                result.update(output["token_stats"])

            all_results.append(result)

    save_results(all_results, args.name, args.model, args.questions)


# 其他函数（load_questions, create_prompt, save_results）保持不变

if __name__ == "__main__":
    main()
