# python3 autoSum.py --target 1 (要汇总的文件编号，只有一个参数)

import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(model_detail_scores, score_num: int):
    """
    绘制模型在9个维度上的比较雷达图
    """

    # 设置维度标签
    labels = [
        "Fact-based - Single choice",
        "Fact-based - Multiple choice",
        "Fact-based - QA",
        "Reading comprehension - Single choice",
        "Reading comprehension - Multiple choice",
        "Reading comprehension - QA",
        "Sentiment analysis - Single choice",
        "Sentiment analysis - Multiple choice",
        "Sentiment analysis - QA",
    ]

    # 设置图形大小（增大图像大小）
    plt.figure(figsize=(12, 12))

    # 计算每个维度的最大得分
    max_scores = [0] * len(labels)
    for scores in model_detail_scores.values():
        for i, score in enumerate(scores):
            if score > max_scores[i]:
                max_scores[i] = score

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形

    # 为每个模型绘制雷达图
    for model_name, scores in model_detail_scores.items():
        # 归一化得分
        normalized_scores = []
        for i, score in enumerate(scores):
            if max_scores[i] == 0:
                normalized_score = 0
            else:
                normalized_score = score / max_scores[i]
            normalized_scores.append(normalized_score)
        normalized_scores = np.concatenate(
            (normalized_scores, [normalized_scores[0]])
        )  # 闭合数据
        plt.polar(angles, normalized_scores, "-o", label=model_name)

    # 设置标签
    plt.xticks(angles[:-1], labels, rotation=45)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title(f"模型在9个维度上的归一化得分比较，评分{score_num}")

    # 保存图片
    plt.savefig(f"model_comparison_{score_num}.png", bbox_inches="tight", dpi=300)
    plt.close()


def read_and_calculate_weighted_score(score_num: int):
    """
    读取评分文件并计算加权总分
    """
    # 文件路径配置
    files = {
        "事实性回答": {
            "单选题": (f"单选题_事实性回答_评分{score_num}.json", 0.30),
            "多选题": (f"多选题_事实性回答_评分{score_num}.json", 0.10),
            "问答题": (f"问答题_事实性回答_评分{score_num}.json", 0.10),
        },
        "基础阅读理解": {
            "单选题": (f"单选题_基础阅读理解_评分{score_num}.json", 0.20),
            "多选题": (f"多选题_基础阅读理解_评分{score_num}.json", 0.12),
            "问答题": (f"问答题_基础阅读理解_评分{score_num}.json", 0.08),
        },
        "情感分析": {
            "单选题": (f"单选题_情感分析_评分{score_num}.json", 0.05),
            "多选题": (f"多选题_情感分析_评分{score_num}.json", 0.02),
            "问答题": (f"问答题_情感分析_评分{score_num}.json", 0.03),
        },
    }

    # 存储每个模型的加权分数和原始分数
    model_scores = defaultdict(float)
    model_detail_scores = defaultdict(lambda: [0] * 9)
    dimension_map = {
        ("事实性回答", "单选题"): 0,
        ("事实性回答", "多选题"): 1,
        ("事实性回答", "问答题"): 2,
        ("基础阅读理解", "单选题"): 3,
        ("基础阅读理解", "多选题"): 4,
        ("基础阅读理解", "问答题"): 5,
        ("情感分析", "单选题"): 6,
        ("情感分析", "多选题"): 7,
        ("情感分析", "问答题"): 8,
    }
    files_processed = 0

    # 遍历每个文件并读取分数
    for problem_type, type_files in files.items():
        for question_type, (filename, weight) in type_files.items():
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    files_processed += 1

                # 存储原始分数和计算加权分数
                dim_idx = dimension_map[(problem_type, question_type)]
                for model_name, model_data in data["model_evaluations"].items():
                    avg_score = model_data["statistics"]["avg_final_score"]
                    model_detail_scores[model_name][dim_idx] = avg_score
                    model_scores[model_name] += avg_score * weight

            except FileNotFoundError:
                print(f"警告: 未找到文件 {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    if files_processed == 0:
        print(f"\n错误: 未找到评分{score_num}的任何文件")
        return

    # 绘制比较图
    plot_comparison(model_detail_scores, score_num)

    # 输出加权分数
    print(f"\n评分{score_num}的最终加权分数:")
    print("-" * 50)
    for model_name, final_score in model_scores.items():
        print(f"{model_name}: {final_score:.2f}")
    print("\n权重说明:")
    print("事实性问答 (50%): 单选题(30%), 多选题(10%), 问答题(10%)")
    print("基础阅读理解 (40%): 单选题(20%), 多选题(12%), 问答题(8%)")
    print("情感分析 (10%): 单选题(5%), 多选题(2%), 问答题(3%)")
    print(f"\n已生成雷达图: model_comparison_{score_num}.png")


def main():
    parser = argparse.ArgumentParser(description="计算模型评分的加权总分")
    parser.add_argument("--target", type=int, required=True, help="评分文件序号")

    args = parser.parse_args()
    read_and_calculate_weighted_score(args.target)


if __name__ == "__main__":
    main()
