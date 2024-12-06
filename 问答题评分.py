import json
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import sys
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


class QAScorer:
    def __init__(self, speed_profile: TokenSpeedProfile):
        self.speed_profile = speed_profile

    def calculate_base_score(self, found_essential: int) -> float:
        """
        基于命中关键词个数的基础分计算

        Args:
            found_essential: 找到的必要关键词数量

        Returns:
            float: 基础分数（满分60分）
        """
        # 定义关键词个数和对应的分数：前部分分值大，保障基础分；后续得分降低，命中太多基础词无法断言模型能力强
        score_map = {
            0: 0,  # 0个关键词 - 0分
            1: 30,  # 1个关键词 - 30分
            2: 40,  # 2个关键词 - 40分
            3: 50,  # 3个关键词 - 50分
            4: 55,  # 4个关键词 - 55分
            5: 60,  # 5个及以上关键词 - 60分
        }

        # 如果命中数超出映射表范围，返回最高分
        if found_essential >= max(score_map.keys()):
            return score_map[max(score_map.keys())]

        # 返回对应分数
        return score_map.get(found_essential, 0)

    def calculate_keyword_bonus(self, matched_count: int) -> float:
        """
        基于命中数的递增奖励计算

        Args:
            matched_count: 匹配到的关键词数量
        Returns:
            float: 奖励分数（满分20分）
        """
        if matched_count == 0:
            return 0

        # 定义命中数和对应的分数
        # 体现关键词的奖励机制，奖励力度逐渐加大。命中多个关键词是比较困难的
        bonus_map = {
            1: 5,  # 1个词给5分
            2: 11,  #
            3: 18,  #
            4: 26,  #
            5: 30,  # 5个及以上词给满分
        }

        # 如果命中数超出映射表范围，返回最高分
        if matched_count >= max(bonus_map.keys()):
            return bonus_map[max(bonus_map.keys())]

        # 返回对应分数
        return bonus_map.get(matched_count, 0)

    def check_length_requirements(
        self, response: str, min_length: int, max_length: int
    ) -> bool:
        """检查回答是否符合长度要求"""
        response_length = len(response.strip())
        return min_length <= response_length <= max_length

    def calculate_accuracy_score(
        self, response: str, reference: dict, status: str, length_valid: bool
    ) -> Dict[str, float]:
        # 其他代码保持不变...

        scores = {"base_score": 0, "keyword_bonus": 0, "detail_bonus": 0, "matches": []}

        # 将回答转换为小写，方便匹配
        response_lower = response.lower()

        # 1. 基础分计算（必要关键词）
        found_essential = 0
        for phrase in reference["essential_keywords"]:
            words = phrase.lower().split()
            for word in words:
                if word in response_lower:
                    found_essential += 1
                    scores["matches"].append(phrase)
                    break  # 匹配到关键词中的任意一个词即可
        scores["base_score"] = self.calculate_base_score(found_essential)
        scores["found_essential_count"] = found_essential
        scores["total_essential_count"] = len(reference["essential_keywords"])

        # 2. 关键词奖励计算（加分关键词）
        found_keywords = 0
        for phrase in reference.get("bonus_keywords", []):
            words = phrase.lower().split()
            for word in words:
                if word in response_lower:
                    found_keywords += 1
                    scores["matches"].append(phrase)
                    break
        scores["keyword_bonus"] = self.calculate_keyword_bonus(found_keywords)

        # 3. 细节加分（细节点）
        detail_found = False
        for phrase in reference.get("detail_points", []):
            words = phrase.lower().split()
            for word in words:
                if word in response_lower:
                    detail_found = True
                    scores["matches"].append(phrase)
                    break  # 找到匹配即可
            if detail_found:
                break  # 已经找到匹配，跳出外层循环
        scores["detail_bonus"] = 10 if detail_found else 0

        # 计算总分并转换为10分制
        total_score = (
            scores["base_score"] + scores["keyword_bonus"] + scores["detail_bonus"]
        )
        accuracy_score = (
            min(100, total_score) / 10
        )  # 确保总分不超过100分，再换算成10分制

        return {
            "accuracy_score": accuracy_score,
            "is_correct": scores["base_score"] >= 48,  # 基础分达到48分（80%）算基本正确
            "partial_scores": scores,
        }

    def calculate_speed_score(self, tokens_per_second: float) -> float:
        """计算速度分数（满分10分）"""
        if tokens_per_second <= 0:
            return 0

        z_score = (
            tokens_per_second - self.speed_profile.baseline_speed
        ) / self.speed_profile.std_dev
        speed_score = 10 / (1 + np.exp(-z_score))
        return min(10.0, max(0.0, speed_score))

    def calculate_score(self, question: Dict) -> Dict[str, float]:
        """计算最终分数"""
        # 如果存在错误，直接返回0分
        if question.get("errors"):
            return {
                "final_score": 0,
                "accuracy_score": 0,
                "speed_score": 0,
                "is_correct": False,
                "partial_scores": {
                    "base_score": 0,
                    "keyword_bonus": 0,
                    "detail_bonus": 0,
                    "matches": [],
                },
                "error": question["errors"],
            }

        # 计算准确率分数
        accuracy_results = self.calculate_accuracy_score(
            response=question["model_output"],
            reference=question["reference"],
            status=question["status"],
            length_valid=question["length_valid"],
        )

        # 计算速度分数
        speed_score = self.calculate_speed_score(question["output_tokens_speed"])

        # 最终得分 = 准确率(90%) + 速度(10%)
        final_score = (accuracy_results["accuracy_score"] * 0.9) + (speed_score * 0.1)

        return {
            "final_score": final_score,
            "accuracy_score": accuracy_results["accuracy_score"],
            "speed_score": speed_score,
            "is_correct": accuracy_results["is_correct"],
            "partial_scores": accuracy_results["partial_scores"],
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

                # 计算平均token速度
                speeds = [
                    r["output_tokens_speed"]
                    for r in data["results"]
                    if "output_tokens_speed" in r and r["status"] == "success"
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
                if "output_tokens_speed" in r and r["status"] == "success"
            ]
            all_speeds.extend(speeds)

        if all_speeds:
            mean_speed = np.mean(all_speeds)
            std_speed = np.std(all_speeds)
            self.speed_stats = TokenSpeedStats(mean_speed, std_speed)
        else:
            print("警告: 没有找到有效的速度数据")
            self.speed_stats = TokenSpeedStats(0.0, 1.0)

    def evaluate_single_model(self, model_result: ModelResult) -> Dict:
        """评估单个模型的表现"""
        scorer = QAScorer(
            TokenSpeedProfile(
                baseline_speed=self.speed_stats.mean, std_dev=self.speed_stats.std
            )
        )

        question_scores = []
        total_score = 0.0
        correct_count = 0
        partial_correct_count = 0
        error_count = 0

        for question in model_result.results:
            score = scorer.calculate_score(question)

            if score.get("error"):
                error_count += 1
            elif score["is_correct"]:
                correct_count += 1
            elif score["accuracy_score"] > 0:
                partial_correct_count += 1

            question_scores.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "reference": question["reference"],
                    "model_output": question.get("model_output", ""),
                    "scores": score,
                    "execution_time": question["execution_time"],
                    "output_tokens_num": question.get("output_tokens_num", 0),
                    "output_tokens_speed": question.get("output_tokens_speed", 0),
                    "length_valid": question["length_valid"],
                    "status": question["status"],
                    "errors": question.get("errors", ""),
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
                "avg_final_score": (
                    total_score / total_questions if total_questions > 0 else 0
                ),
                "full_correct_rate": (
                    correct_count / total_questions if total_questions > 0 else 0
                ),
                "partial_correct_rate": (
                    partial_correct_count / total_questions
                    if total_questions > 0
                    else 0
                ),
                "error_rate": (
                    error_count / total_questions if total_questions > 0 else 0
                ),
                "avg_token_speed": model_result.avg_token_speed,
                "total_questions": total_questions,
            },
            "question_scores": question_scores,
        }

    def evaluate_all_models(self) -> Dict:
        """评估所有模型的表现"""
        if not self.speed_stats:
            self.calculate_speed_statistics()

        all_evaluations = {}
        for (executable, model), result in self.model_results.items():
            model_key = f"{executable}_{model}"
            evaluation = self.evaluate_single_model(result)
            all_evaluations[model_key] = evaluation

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
        if question_type != "问答题":
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
        filename = f"问答题_{problem_type}_评分{index}.json"
        if not os.path.exists(filename):
            return filename
        index += 1


def main():
    parser = argparse.ArgumentParser(description="评估多个模型的问答题测试结果")
    parser.add_argument(
        "--jsons", nargs="+", required=True, help="要评估的JSON文件列表"
    )
    args = parser.parse_args()

    # 验证文件名
    is_valid, invalid_files, error_message = validate_all_files(args.jsons)

    if not is_valid:
        print("错误: 输入文件验证失败")
        print("\n文件名格式要求:")
        print("1. name_model_问答题_问题type_问题集.json")
        print("2. name_问答题_问题type_问题集.json")
        print("\n要求:")
        print('- 题型必须为: "问答题"')
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

    # 评估处理
    evaluator = MultiModelEvaluator()
    evaluator.load_result_files(args.jsons)
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
