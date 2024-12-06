# 用法：python3 问答题自动测试.py --name *** (--model ***) --questions ***.json

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
    """创建问答题prompt"""
    prompt = f"{question_data['question']}\n"
    # 添加长度要求提示
    prompt += f"请在{question_data['min_length']}到{question_data['max_length']}字之间回答问题。"
    return prompt


def validate_answer_length(answer: str, min_length: int, max_length: int) -> bool:
    """验证答案长度是否符合要求"""
    answer_length = len(answer)
    return min_length <= answer_length <= max_length


def save_results(results, executable_name, model_number, questions_file):
    """保存测试结果到JSON文件"""
    questions_basename = os.path.splitext(os.path.basename(questions_file))[0]
    model_part = f"_{model_number}" if model_number else ""
    output_filename = f"{executable_name}{model_part}_{questions_basename}.json"

    output_data = {
        "test_info": {
            "executable": executable_name,
            "model": model_number,
            "questions_file": questions_file,
            "timestamp": datetime.now().isoformat(),
            "test_type": "qa",  # 添加测试类型标识
        },
        "results": results,
    }

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_filename}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def extract_token_stats(stdout):
    """提取token统计信息"""
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
    lines = stdout.split("\n")
    debug_patterns = [
        r"avx:.+",
        r"temp:.+",
        r"retrieved the files.+",
        r"loaded the model.+",
        r"\d+ tokens generated.+",
    ]
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
    parser = argparse.ArgumentParser(description="执行可执行文件并传入问答题")
    parser.add_argument("--name", type=str, required=True, help="可执行文件名")
    parser.add_argument("--model", type=str, required=False, help="模型编号")
    parser.add_argument(
        "--questions",
        type=str,
        default="问答题数据集.json",
        help="问答题数据集JSON文件路径",
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
            # 清理model输出，移除问题重复和提示语
            raw_output = output["stdout"]
            quote_pattern = f'"{question_data["question"]}.*?问题。"\\s*'
            cleaned_output = re.sub(quote_pattern, "", raw_output, flags=re.DOTALL)

            result = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "reference": question_data["reference"],  # 保存参考答案信息
                "min_length": question_data["min_length"],
                "max_length": question_data["max_length"],
                "execution_time": datetime.now().isoformat(),
                "model_output": cleaned_output.strip(),  # 使用清理后的输出
                "errors": output["stderr"],
                "status": "success" if output["return_code"] == 0 else "failed",
            }

            # 验证答案长度
            result["length_valid"] = validate_answer_length(
                output["stdout"],
                question_data["min_length"],
                question_data["max_length"],
            )

            # 添加token统计信息
            if output["token_stats"]:
                result.update(output["token_stats"])

            all_results.append(result)

    save_results(all_results, args.name, args.model, args.questions)


if __name__ == "__main__":
    main()
