import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局配置：设置中文字体和图表清晰度
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 (Windows自带黑体)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 300  # 设置导出图片的高清分辨率 (300 dpi 适合打印)

# ==========================================
# 2. 数据录入 (根据你提供的真实数据)
# ==========================================
x_labels = ['0%', '20%', '50%', '100%']

# 数据格式: [完整数据(0%), 缺失20%, 缺失50%, 缺失100%]
data = {
    "Baseline": {
        "text": [81.31, 76.27, 68.50, 34.24],
        "image": [81.31, 80.86, 80.03, 77.33]
    },
    "TGA (missing_rate=0.0)": {
        "text": [82.54, 74.41, 65.39, 33.97],
        "image": [82.54, 75.33, 65.24, 35.62]
    },
    "EANN (missing_rate=0.0)": {
        "text": [82.85, 76.88, 66.65, 35.17],
        "image": [82.85, 81.96, 81.40, 79.94]
    },
    "EANN (missing_rate=0.2)": {
        "text": [82.55, 79.62, 74.52, 67.23],
        "image": [82.55, 82.47, 80.97, 80.07]
    },
    "EANN (missing_rate=0.5)": {
        "text": [82.02, 80.00, 75.72, 69.52],
        "image": [82.02, 81.81, 81.42, 80.86]
    },
    "EANN (missing_rate=1.0)": {
        "text": [81.38, 78.82, 75.70, 69.91],
        "image": [81.38, 81.05, 81.40, 80.84]
    },
    "EANN+KD (missing_rate=0.0)": {
        "text": [82.86, 78.53, 71.02, 58.46],
        "image": [82.86, 82.34, 81.03, 79.40]
    },
    "EANN+KD (missing_rate=0.5)": {
        "text": [83.24, 79.83, 75.61, 65.92],
        "image": [83.24, 82.62, 81.89, 80.18]
    },
    "EANN+KD (missing_rate=1.0)": {
        "text": [81.40, 79.34, 75.50, 69.33],
        "image": [81.40, 81.10, 81.14, 81.14]
    },
    "TGA+KD (missing_rate=0.0)": {
        "text": [83.16, 77.03, 66.46, 49.94],
        "image": [83.16, 76.28, 67.85, 50.26]
    },
    "TGA+KD (missing_rate=0.2)": {
        "text": [81.27, 78.46, 75.53, 69.10],
        "image": [81.27, 80.71, 80.20, 78.29]
    },
    "TGA+KD (missing_rate=0.5)": {
        "text": [84.29, 81.81, 77.39, 69.47],
        "image": [84.29, 83.61, 82.47, 81.18]
    },
    "TGA+KD (missing_rate=1.0)": {
        "text": [77.86, 75.85, 73.70, 69.33],
        "image": [77.86, 78.76, 80.07, 82.25]
    }
}

# ==========================================
# 3. 绘图辅助函数
# ==========================================
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def plot_comparison(models_to_plot, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, model_name in enumerate(models_to_plot):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 绘制左图：文本缺失
        ax1.plot(x_labels, data[model_name]["text"], marker=marker, linewidth=2,
                 markersize=8, color=color, label=model_name)

        # 绘制右图：图像缺失
        ax2.plot(x_labels, data[model_name]["image"], marker=marker, linewidth=2,
                 markersize=8, color=color, label=model_name)

    # 设置左图属性
    ax1.set_title('文本模态缺失场景', fontsize=14, pad=15)
    ax1.set_xlabel('文本缺失率', fontsize=12)
    ax1.set_ylabel('准确率 (Accuracy, %)', fontsize=12)
    ax1.set_ylim(30, 90)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10, loc='lower left')

    # 设置右图属性
    ax2.set_title('视觉模态缺失场景', fontsize=14, pad=15)
    ax2.set_xlabel('图像缺失率', fontsize=12)
    ax2.set_ylabel('准确率 (Accuracy, %)', fontsize=12)
    ax2.set_ylim(30, 90)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10, loc='lower left')

    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.close()
    print(f"✅ 成功生成图表: {filename}.png")


# ==========================================
# 4. 生成 4 张对比图
# ==========================================

# 图1：基准、EANN(四种缺失率训练)、TGA(0.0)
plot_1_models = [
    "Baseline", "TGA (missing_rate=0.0)", "EANN (missing_rate=0.0)",
    "EANN (missing_rate=0.2)", "EANN (missing_rate=0.5)", "EANN (missing_rate=1.0)"
]
plot_comparison(plot_1_models,
                "图1：传统基准、基础TGA与EANN多缺失率训练模型性能对比",
                "Figure_1_Baseline_EANN_TGA")

# 图2：EANN+KD (三种缺失率训练)
plot_2_models = ["EANN+KD (missing_rate=0.0)", "EANN+KD (missing_rate=0.5)", "EANN+KD (missing_rate=1.0)"]
plot_comparison(plot_2_models,
                "图2：EANN+KD 知识蒸馏在不同训练缺失率下的性能演变",
                "Figure_2_EANN_KD_Comparison")

# 图3：TGA+KD (四种缺失率训练)
plot_3_models = ["TGA+KD (missing_rate=0.0)", "TGA+KD (missing_rate=0.2)", "TGA+KD (missing_rate=0.5)", "TGA+KD (missing_rate=1.0)"]
plot_comparison(plot_3_models,
                "图3：TGA-KD 知识蒸馏框架在不同训练缺失率下的性能演变",
                "Figure_3_TGA_KD_Comparison")

# 图4：最佳模型终极巅峰对决
plot_4_models = [
    "Baseline",
    "EANN (missing_rate=0.0)",
    "TGA (missing_rate=0.0)",
    "EANN+KD (missing_rate=0.5)",
    "TGA+KD (missing_rate=0.5)"
]
plot_comparison(plot_4_models,
                "图4：面向模态缺失场景各架构最佳模型终极性能对比 (消融实验)",
                "Figure_4_Best_Models_Ultimate_Comparison")

print("\n🎉 所有图表生成完毕！请在当前目录下查看这 4 张 PNG 图片。")