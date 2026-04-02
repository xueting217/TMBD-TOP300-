"""
TMBD-TOP300电影榜单数据统计分析
功能：读取电影数据，生成多维度统计图表
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pandas import Series


def setup_matplotlib():
    """配置 matplotlib 中文显示和默认参数"""
    plt.rcParams['font.sans-serif'] = ['SimHei']


def load_movie_data(filepath: str) -> pd.DataFrame:
    """
    加载电影数据
    
    Args:
        filepath: CSV 文件路径
        
    Returns:
        DataFrame: 包含电影数据的 DataFrame
    """
    return pd.read_csv(
        filepath,
        usecols=['电影名', '年份', '上映时间', '类型', '时长', '评分', '语言'],
        dtype={'年份': 'Int64'}
    )


def preprocess_year_data(data: pd.DataFrame) -> None:
    """
    处理年份数据的缺失值
    
    Args:
        data: 电影数据 DataFrame
    """
    # 用上映时间的前 4 位填充缺失的年份
    data['年份'] = data['年份'].fillna(data['上映时间'].str[0:4])


def plot_yearly_movie_count(data: pd.DataFrame, axes: Axes) -> None:
    """
    绘制每年电影数量变化折线图
    
    Args:
        data: 电影数据 DataFrame
        axes: matplotlib Axes 对象
    """
    # 分组统计每年的电影数量
    year_count = data.groupby('年份')['年份'].count()
    
    # 准备数据
    min_year = year_count.index.min()
    max_year = year_count.index.max()
    x = list(range(min_year, max_year + 1))
    y = [int(year_count.get(i, 0)) for i in x]
    
    # 绘制折线图
    axes.plot(x, y, color='red', linewidth=2)
    axes.plot(x, y, color='green')
    
    # 设置标题和标签
    axes.set_title('每年电影数量变化折线图', fontsize=15)
    axes.set_xlabel('年份', fontsize=12)
    axes.set_ylabel('电影数量', fontsize=12)
    
    # 设置刻度间隔
    axes.set_xticks(x[::8])
    y_ticks = list(range(0, 31, 3))
    axes.set_yticks(y_ticks)
    
    # 添加网格
    axes.grid(linestyle='--', alpha=0.5)


def plot_language_distribution(data: pd.DataFrame, axes: Axes) -> None:
    """
    绘制不同语言电影数量柱状图
    
    Args:
        data: 电影数据 DataFrame
        axes: matplotlib Axes 对象
    """
    # 统计各语言的电影数量
    language_count = data.groupby('语言')['语言'].count().sort_values(ascending=False)
    
    x_language = language_count.index.tolist()
    y_language = language_count.values.tolist()
    
    # 绘制柱状图
    axes.bar(x_language, y_language, color='green', width=0.7)
    
    # 设置标题和标签
    axes.set_title('不同语言电影数量柱状图', fontsize=15)
    axes.set_xlabel('语言', fontsize=12)
    axes.set_ylabel('电影数量', fontsize=12)
    
    # 添加网格和旋转 x 轴标签
    axes.grid(linestyle='--', alpha=0.5)
    axes.tick_params(axis='x', rotation=90)


def count_movie_types(types_series: pd.Series) -> dict:
    """
    统计各类型电影的数量（支持多类型）
    
    Args:
        types_series: 包含类型字符串的 Series（逗号分隔）
        
    Returns:
        dict: 类型计数字典
    """
    type_count = {}
    for types in types_series.str.split(','):
        for movie_type in types:
            type_count[movie_type] = type_count.get(movie_type, 0) + 1
    return type_count


def plot_type_distribution(data: pd.DataFrame, axes: Axes) -> None:
    """
    绘制不同类型电影数量柱状图
    
    Args:
        data: 电影数据 DataFrame
        axes: matplotlib Axes 对象
    """
    # 统计各类型的电影数量
    type_count = count_movie_types(data['类型'])
    
    x_types = list(type_count.keys())
    y_types = list(type_count.values())
    
    # 绘制柱状图
    axes.bar(x_types, y_types, color='green', width=0.7)
    
    # 设置标题和标签
    axes.set_title('不同类型电影数量柱状图', fontsize=15)
    axes.set_xlabel('类型', fontsize=12)
    axes.set_ylabel('电影数量', fontsize=12)
    
    # 添加网格和旋转 x 轴标签
    axes.grid(linestyle='--', alpha=0.5)
    axes.tick_params(axis='x', rotation=90)


def merge_small_categories(score_count: Series, threshold: float = 0.02) -> Series:
    """
    合并占比小于阈值的评分类别
    
    Args:
        score_count: 评分计数 Series
        threshold: 最小占比阈值，默认 2%
        
    Returns:
        Series: 合并后的评分计数
    """
    total = score_count.sum()
    large_scores = score_count.loc[score_count >= total * threshold]
    small_scores = score_count.loc[score_count < total * threshold]
    
    if small_scores.shape[0] > 0:
        large_scores['其他'] = small_scores.sum()
    
    return large_scores


def plot_score_distribution(data: pd.DataFrame, axes: Axes) -> None:
    """
    绘制不同评分电影数量占比饼状图
    
    Args:
        data: 电影数据 DataFrame
        axes: matplotlib Axes 对象
    """
    # 统计各评分的电影数量
    score_count = data.groupby('评分')['评分'].count()
    
    # 合并小数据
    large_scores = merge_small_categories(score_count)
    
    x_score = large_scores.index.tolist()
    y_score = large_scores.values.tolist()
    
    # 绘制饼状图
    axes.pie(
        y_score,
        labels=x_score,
        autopct='%1.1f%%',
        startangle=0,
        radius=1.2
    )
    
    # 设置标题和图例
    axes.set_title('不同评分电影数量占比饼状图', fontsize=15)
    axes.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3))


def create_subplots(nrows: int = 2, ncols: int = 2, 
                    figsize: tuple = (20, 12)) -> tuple:
    """
    创建子图布局
    
    Args:
        nrows: 行数
        ncols: 列数
        figsize: 画布大小
        
    Returns:
        tuple: (fig, axes 数组)
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle('TMBD-TOP300电影榜单数据统计', fontsize=23, x=0.5, y=0.98)
    fig.subplots_adjust(wspace=0.5, hspace=0.4)
    return fig, axes


def save_and_show_plot(fig: plt.Figure, output_path: str) -> None:
    """
    保存并显示图表
    
    Args:
        fig: matplotlib Figure 对象
        output_path: 输出文件路径
    """
    fig.savefig(output_path)
    plt.show()


def main():
    """主函数：执行完整的数据分析流程"""
    # 配置 matplotlib
    setup_matplotlib()
    
    # 创建子图
    fig, axes = create_subplots()
    axes1: Axes = axes[0, 0]
    axes2: Axes = axes[0, 1]
    axes3: Axes = axes[1, 0]
    axes4: Axes = axes[1, 1]
    
    # 加载数据
    data = load_movie_data('data/movies.csv')
    
    # 数据预处理
    preprocess_year_data(data)
    
    # 生成四个图表
    plot_yearly_movie_count(data, axes1)
    plot_language_distribution(data, axes2)
    plot_type_distribution(data, axes3)
    plot_score_distribution(data, axes4)
    
    # 保存并显示
    save_and_show_plot(fig, 'data/test.png')


if __name__ == '__main__':
    main()
