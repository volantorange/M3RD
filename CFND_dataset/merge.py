import pandas as pd
import sys


def merge_csv_files(file_paths, output_path):
    """
    合并多个具有相同列结构的 CSV 文件
    参数:
        file_paths: list, 输入的 CSV 文件路径列表
        output_path: str, 合并后的输出文件路径
    """
    dfs = []
    expected_columns = ['num', 'title', 'imag', 'label', 'event']

    for i, path in enumerate(file_paths):
        try:
            df = pd.read_csv(path, encoding='utf-8')
            # 检查列名是否匹配
            if list(df.columns) != expected_columns:
                print(
                    f"警告: 文件 {path} 的列名 {list(df.columns)} 与期望的 {expected_columns} 不一致，仍将合并但可能导致错位")
            dfs.append(df)
            print(f"已读取: {path} (行数: {len(df)})")
        except Exception as e:
            print(f"读取文件 {path} 出错: {e}")
            continue

    if not dfs:
        print("没有成功读取任何文件，退出。")
        return

    # 纵向拼接
    merged_df = pd.concat(dfs, ignore_index=True)

    # 保存
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n合并完成！总行数: {len(merged_df)}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    # 请根据实际情况修改这三个文件的路径
    file1 = "test_data.csv"  # 替换为你的第一个 CSV 路径
    file2 = "train_data.csv"  # 替换为你的第二个 CSV 路径
    file3 = "val_data.csv"  # 替换为你的第三个 CSV 路径
    output = "merged.csv"  # 合并后的输出文件名

    # 如果要在命令行传参，可以取消下面三行注释，并注释掉上面的固定路径
    # if len(sys.argv) < 4:
    #     print("用法: python merge_csv.py <文件1> <文件2> <文件3> [输出文件]")
    #     sys.exit(1)
    # file1, file2, file3 = sys.argv[1:4]
    # output = sys.argv[4] if len(sys.argv) > 4 else "merged.csv"

    merge_csv_files([file1, file2, file3], output)