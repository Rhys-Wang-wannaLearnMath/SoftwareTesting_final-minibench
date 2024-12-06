# 用法：python3 多选题评分.py --jsons 1.json 2.json 3.json

import sys
import json
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class ModelResult:
    """存储单个模型的测试结果"""

    executable: str
    model: str
    questions_file: str
    results: List[Dict]
    avg_token_speed: float


@dataclass
class TokenSpeedStats:
    """Token速度统计信息"""

    mean: float
    std: float


@dataclass
class TokenSpeedProfile:
    """Token速度基准配置"""

    baseline_speed: float
    std_dev: float


class MultiChoiceScorer:
    def __init__(self, speed_profile: TokenSpeedProfile):
        self.speed_profile = speed_profile

    def evaluate_multi_choice(
        self, model_answers: list, correct_answers: list, total_options=5
    ):
        """多选题评分算法"""
        # 转换为集合
        model_set = set(model_answers)
        correct_set = set(correct_answers)

        # 基础计算
        true_positives = len(model_set & correct_set)
        false_positives = len(model_set - correct_set)
        false_negatives = len(correct_set - model_set)

        # 1. 基础分（60分）
        recall = true_positives / len(correct_set)
        base_score = 60 * recall

        # 2. 精确度分数（20分）
        precision = true_positives / len(model_set) if model_set else 0
        precision_score = 20 * precision

        # 3. 完整性分数（20分）
        completeness = 1.0 - abs(len(model_set) - len(correct_set)) / len(correct_set)
        completeness_score = 20 * max(0, completeness)

        # 转换为10分制
        total_score = (base_score + precision_score + completeness_score) / 10

        return total_score, {
            "correct_selections": true_positives,
            "wrong_selections": false_positives,
            "missed_selections": false_negatives,
            "base_score": round(base_score, 2),
            "precision_score": round(precision_score, 2),
            "completeness_score": round(completeness_score, 2),
        }

    def calculate_score(
        self, model_output: str, correct_answers: list, tokens_per_second: float
    ) -> Dict[str, float]:
        """计算最终分数"""
        # 将模型输出转换为答案列表
        model_answers = list(model_output.strip().upper())

        # 计算准确率得分（满分10分）
        accuracy_score, details = self.evaluate_multi_choice(
            model_answers, correct_answers
        )

        # 计算速度得分（满分10分）
        z_score = (
            tokens_per_second - self.speed_profile.baseline_speed
        ) / self.speed_profile.std_dev
        speed_score = 10 / (1 + np.exp(-z_score))
        speed_score = min(10.0, max(0.0, speed_score))

        # 最终得分 = 准确率(90%) + 速度(10%)
        final_score = (accuracy_score * 0.9) + (speed_score * 0.1)

        return {
            "final_score": round(final_score, 2),
            "accuracy_score": round(accuracy_score, 2),
            "speed_score": round(speed_score, 2),
            "scoring_details": details,
        }


class MultiModelEvaluator:
    def __init__(self):
        self.model_results: Dict[Tuple[str, str], ModelResult] = {}
        self.speed_stats = None

    def load_result_files(self, json_files: List[str]):
        """加载指定的结果JSON文件"""
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                test_info = data["test_info"]
                executable = test_info["executable"]
                model = test_info.get("model", "default")

                speeds = [
                    r["output_tokens_speed"]
                    for r in data["results"]
                    if "output_tokens_speed" in r
                ]
                avg_speed = np.mean(speeds) if speeds else 0.0

                key = (executable, model)
                self.model_results[key] = ModelResult(
                    executable=executable,
                    model=model,
                    questions_file=test_info["questions_file"],
                    results=data["results"],
                    avg_token_speed=avg_speed,
                )
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

    def calculate_speed_statistics(self):
        """计算所有模型的token速度统计信息"""
        all_speeds = []
        for result in self.model_results.values():
            speeds = [
                r["output_tokens_speed"]
                for r in result.results
                if "output_tokens_speed" in r
            ]
            all_speeds.extend(speeds)

        if all_speeds:
            self.speed_stats = TokenSpeedStats(np.mean(all_speeds), np.std(all_speeds))
        else:
            print("警告: 没有找到有效的速度数据")

    def evaluate_single_model(self, model_result: ModelResult) -> Dict:
        """评估单个模型的表现"""
        scorer = MultiChoiceScorer(
            TokenSpeedProfile(
                baseline_speed=self.speed_stats.mean, std_dev=self.speed_stats.std
            )
        )

        question_scores = []
        total_score = 0.0

        for question in model_result.results:
            score = scorer.calculate_score(
                model_output=question["model_output"],
                correct_answers=question["correct_answers"],
                tokens_per_second=question["output_tokens_speed"],
            )

            question_scores.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answers": question["correct_answers"],
                    "model_answers": list(question["model_output"].strip().upper()),
                    "scores": score,
                    "execution_time": question["execution_time"],
                    "output_tokens_num": question["output_tokens_num"],
                    "output_tokens_speed": question["output_tokens_speed"],
                }
            )

            total_score += score["final_score"]

        total_questions = len(model_result.results)
        return {
            "model_info": {
                "executable": model_result.executable,
                "model": model_result.model,
                "questions_file": model_result.questions_file,
            },
            "statistics": {
                "avg_final_score": round(
                    total_score / total_questions if total_questions > 0 else 0, 2
                ),
                "avg_token_speed": model_result.avg_token_speed,
                "total_questions": total_questions,
            },
            "question_scores": question_scores,
        }

    def evaluate_all_models(self) -> Dict:
        """评估所有模型的表现"""
        if not self.speed_stats:
            print("警告: 未计算速度统计信息")
            return {}

        all_evaluations = {}
        for (executable, model), result in self.model_results.items():
            model_key = f"{executable}_{model}"
            all_evaluations[model_key] = self.evaluate_single_model(result)

        return {
            "speed_statistics": {
                "mean": self.speed_stats.mean,
                "std": self.speed_stats.std,
            },
            "model_evaluations": all_evaluations,
        }


def validate_filename(filename: str) -> Tuple[bool, str, str]:
    """
    验证文件名是否符合规范，并返回题型和问题类型
    返回: (是否有效, 题型, 问题类型)
    """
    try:
        # 去除.json后缀并分割文件名
        parts = filename.replace(".json", "").split("_")

        # 检查文件名部分数量（4或5个部分）
        if len(parts) not in [5, 6]:
            return False, "", ""

        # 提取题型和问题类型
        if len(parts) == 6:  # 带model的情况
            _, _, _, question_type, problem_type, _ = parts
        else:  # 不带model的情况
            _, _, question_type, problem_type, _ = parts

        # 验证题型
        if question_type != "多选题":
            return False, "", ""

        # 验证问题类型
        valid_problem_types = ["事实性回答", "基础阅读理解", "情感分析"]
        if problem_type not in valid_problem_types:
            return False, "", ""

        return True, question_type, problem_type
    except:
        return False, "", ""


def validate_all_files(filenames: List[str]) -> Tuple[bool, List[str], str]:
    """
    验证所有文件名并确保问题类型一致
    返回: (是否全部有效, 无效文件列表, 错误信息)
    """
    if not filenames:
        return False, [], "未提供文件"

    invalid_files = []
    problem_types = set()

    for filename in filenames:
        is_valid, _, problem_type = validate_filename(filename)
        if not is_valid:
            invalid_files.append(filename)
        else:
            problem_types.add(problem_type)

    # 检查是否有无效文件
    if invalid_files:
        return False, invalid_files, "文件名格式不符合规范"

    # 检查问题类型是否一致
    if len(problem_types) > 1:
        return False, [], f"提供的文件包含不同的问题类型: {', '.join(problem_types)}"

    return True, [], ""


def get_available_filename(problem_type: str) -> str:
    """获取可用的结果文件名"""
    index = 1
    while True:
        filename = f"多选题_{problem_type}_评分{index}.json"
        if not os.path.exists(filename):
            return filename
        index += 1


def main():
    parser = argparse.ArgumentParser(description="评估多个模型的多选题测试结果")
    parser.add_argument(
        "--jsons", nargs="+", required=True, help="要评估的JSON文件列表"
    )
    args = parser.parse_args()

    # 验证文件名
    is_valid, invalid_files, error_message = validate_all_files(args.jsons)

    if not is_valid:
        print("错误: 输入文件验证失败")
        print("\n文件名格式要求:")
        print("1. name_model_多选题_问题type_问题集.json")
        print("2. name_多选题_问题type_问题集.json")
        print("\n要求:")
        print('- 题型必须为: "多选题"')
        print('- 问题type必须为: "事实性回答"、"基础阅读理解"或"情感分析"')
        print("- 一次只能处理相同问题type的文件")

        if invalid_files:
            print("\n不合规的文件:")
            for f in invalid_files:
                print(f"- {f}")
        else:
            print(f"\n错误原因: {error_message}")

        sys.exit(1)

    # 获取问题类型（已经验证过所有文件的问题类型相同）
    _, _, problem_type = validate_filename(args.jsons[0])

    # 初始化评估器并处理
    evaluator = MultiModelEvaluator()
    evaluator.load_result_files(args.jsons)
    evaluator.calculate_speed_statistics()
    results = evaluator.evaluate_all_models()

    # 获取带编号的输出文件名
    output_filename = get_available_filename(problem_type)

    # 保存结果
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存到: {output_filename}")
    except Exception as e:
        print(f"保存结果时出错: {e}")


if __name__ == "__main__":
    main()
