import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from fitter import Fitter


WORK_PATH = "./"
os.chdir(WORK_PATH)
# print("WORK_PATH:", os.getcwd())


class Classify:
    def __init__(self, index, columns, df, worker):
        self.manager = mp.Manager
        self.same_group_data = self.manager().list()  # share list in mp
        self.diff_group_data = self.manager().list()
        self.count = self.manager().Value('i', 0)
        self.index = index
        self.columns = columns
        self.df = df
        self.worker = worker

    def classify(self, idx):
        self.count.value += 1
        if self.count.value % 10 == 0:
            print(f"[{get_time()}] count = {self.count.value} pid = {os.getpid()}")
        same_label_data = self.df.loc[[idx], [idx]].to_numpy()
        self.same_group_data.extend(same_label_data)
        for col in self.columns:
            if idx != col:
                diff_label_data = self.df.loc[[idx], [col]].to_numpy()
                self.diff_group_data.extend(diff_label_data)

    def flow(self):
        print(f"[{get_time()}] worker = {self.worker}")
        print(f"[{get_time()}] total = {len(self.index)}")
        print(f"[{get_time()}] count = 0  ppid = {os.getpid()}")
        pool = mp.Pool(self.worker)
        # progress = tqdm(total=len(self.index))
        for idx in self.index:
            # progress.update(1)
            pool.apply_async(self.classify, args=(idx,))

        pool.close()
        pool.join()


def read(csv_path, dataset, metric):
    print(f"[{get_time()}] Start reading file.")
    # 分块读, 避免卡死
    df_chunk = pd.read_csv(csv_path[0], index_col=0, on_bad_lines='skip', chunksize=100000)
    chunk_list = [] 
    for chunk in df_chunk:
        chunk_list.append(chunk)
    df1 = pd.concat(chunk_list)  # 再把这些块组合成一个DataFrame
    print(f"{df1.shape=}, {csv_path[0]} Done! ")
    
    df2 = pd.read_csv(csv_path[1])
    print(f"{df2.shape=}, {csv_path[1]} Done! ")
    
    drop_list = df2[df2['videoID'] != df2['label']].index.tolist()
    df = df1.drop(index=df1.index[drop_list])  # 剔除识别错误的样本
    print(f"{df.shape=}, Drop Done! ")
    print()
    df.index = df.index.map(lambda x: x.split('-')[0])
    df.columns = df.columns.map(lambda x: x.split('-')[0])
    del df1
    del df2

    # print(df)
    print(f"[{get_time()}] Extract index.")
    index = sorted(set(df.index.values.tolist()))
    columns = sorted(set(df.columns.values.tolist()))

    print(f"[{get_time()}] Classify data.")
    op = Classify(index, columns, df, worker=8)
    op.flow()
    import itertools
    same_group_data = np.array(list(itertools.chain.from_iterable(op.same_group_data)))
    same_group_data = same_group_data[~np.isnan(same_group_data)]
    diff_group_data = np.array(list(itertools.chain.from_iterable(op.diff_group_data)))
    diff_group_data = diff_group_data[~np.isnan(diff_group_data)]
    del op

    # 数据多的时候处理太慢
    # print(f"[{get_time()}] Classify data.")
    # same_group_data, diff_group_data = np.array([]), np.array([])
    # progress = tqdm(total=len(index))
    # for idx in index:
    #     same_label_data = df.loc[[idx], [idx]].to_numpy()
    #     same_group_data = np.append(same_group_data, same_label_data)
    #     for col in columns:
    #         if idx != col:
    #             diff_label_data = df.loc[[idx], [col]].to_numpy()
    #             diff_group_data = np.append(diff_group_data, diff_label_data)
    #     progress.update(1)

    print()
    print(f"{dataset=}, total: {len(same_group_data) + len(diff_group_data)}, {len(same_group_data)=}, {len(diff_group_data)=}")

    # count
    print()
    print(f"[{get_time()}] Count.")
    
    scale = 0.1 if metric == "euc" else 0.01
    
    path = os.path.join("dist_result/" + dataset + "/" + dataset + "-")
    os.makedirs(os.path.join("dist_result", dataset), exist_ok=True)
    
    value_count(same_group_data, scale, path)
    value_count(diff_group_data, scale, path)
    print()
    get_p_value(same_group_data, diff_group_data)

    # draw
    print()
    print(f"[{get_time()}] Draw.")
    same_group_save_path = path + str(len(same_group_data)) 
    diff_group_save_path = path + str(len(diff_group_data)) 
    draw_distribution_histogram(same_group_data, save_path=same_group_save_path + "-same-group-density-distribution.jpg",
                                xlabel="Dist", ylabel="Density", title=dataset + " Same Group Density Distribution",
                                stat="density", binwidth=scale, binrange=None, is_kde=True, is_cumulative=False)
    draw_distribution_histogram(same_group_data, save_path=same_group_save_path +  "-same-group-count-distribution.jpg",
                                xlabel="Dist", ylabel="Count", title=dataset + " Same Group Count Distribution",
                                stat="count", binwidth=scale, binrange=None, is_kde=False, is_cumulative=False)
    draw_distribution_histogram(same_group_data, save_path=same_group_save_path + "-same-group-cumulative-distribution.jpg",
                                xlabel="Dist", ylabel="Cumulative Probability", title=dataset + " Same Group Cumulative Distribution",
                                stat="density", binwidth=scale, binrange=None, is_kde=True, is_cumulative=True)

    draw_distribution_histogram(diff_group_data, save_path=diff_group_save_path + "-diff-group-density-distribution.jpg",
                                xlabel="Dist", ylabel="Density", title=dataset + " Diff Group Density Distribution",
                                stat="density", binwidth=scale, binrange=None, is_kde=True, is_cumulative=False)
    draw_distribution_histogram(diff_group_data, save_path=diff_group_save_path + "-diff-group-count-distribution.jpg",
                                xlabel="Dist", ylabel="Count", title=dataset + " Diff Group Count Distribution",
                                stat="count", binwidth=scale, binrange=None, is_kde=False, is_cumulative=False)
    draw_distribution_histogram(diff_group_data, save_path=diff_group_save_path + "-diff-group-cumulative-distribution.jpg",
                                xlabel="Dist", ylabel="Cumulative Probability", title=dataset + " Diff Group Cumulative Distribution",
                                stat="density", binwidth=scale, binrange=None, is_kde=True, is_cumulative=True)
                                
    # fit
    print()
    print(f"[{get_time()}] Fit.")
    print(f"[{get_time()}] same_group:")
    data_fit(same_group_data, path)
    print()
    print(f"[{get_time()}] diff_group:")
    data_fit(diff_group_data, path)
    print(f"[{get_time()}] Done!")


def get_time():
    return datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')


def value_count(arrA, scale, path):

    a = np.array(arrA)
    res = pd.Series([0])

    ceil = np.round((np.ceil(np.max(a) / scale) + 1) * scale, 2)
    range_list = np.round(np.arange(0., ceil, scale), 2)

    for i in range_list:
        j = np.round(i + scale, 2)
        s_bool = ((a >= i) & (a < j))
        res[j] = s_bool.sum()

    save_path = os.path.join(path + "value_count-" + str(res.sum()) + ".csv")
    res.to_csv(save_path, sep=',')
    # print("Value count result saved to {}/{}".format(os.getcwd(), save_path))

    print(res.sum(), res.to_dict())


def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    print(f"same_group: {np.mean(a)=}, {np.var(a)=}, {np.std(a)=}")
    print(f"diff_group: {np.mean(b)=}, {np.var(b)=}, {np.std(b)=}")

    print(stats.levene(arrA, arrB))  # 检验结果为p>0.05，则可以认为方差是相等的
    t, p = stats.ttest_ind(a, b, equal_var=False)  # equal_var： 如果为 True(默认)，则执行假设总体方差相等的标准独立 2 样本检验。如果为 False，请执行 Welch 的 t-test，它不假设人口方差相等。
    print(f"T检验 : {t=}, {p=}")
    return p


def data_fit(data, path):
    # distributions = None  # 默认尝试80种分布
    distributions = ["norm", "t", "laplace", "cauchy", "chi2", "expon", "exponpow", "gamma", "lognorm", "powerlaw", "rayleigh", "uniform", "johnsonsb", "gausshyper", "beta", "genextreme", "weibull_max", "geninvgauss", "genhyperbolic", "logistic", "gumbel_r", "burr", "mielke", "nct"]
    f = Fitter(data, distributions=distributions, timeout=10000)  # 创建Fitter类
    f.fit()  # 调用fit函数拟合分布
    
    f.summary()  # 输出拟合结果
    save_path = str(len(data)) + ".jpg"
    plt.savefig(path + "f-summary-" + save_path, dpi=300)
    plt.show()
    plt.cla()
    print(f"f.summary():")
    print(f"{f.summary()}")
    
    print()
    print(f"{f.get_best()=}")  # 最佳分布及其参数
    
    print()
    print(f"f.df_errors:")
    print(f"{f.df_errors}")
    print(f"{f.fitted_param['logistic']=}")  # 指定"logistic"分布的参数
    print(f"{f.fitted_param['t']=}")  # 指定"t"分布的参数
    print(f"{f.fitted_param['johnsonsb']=}")  # 指定"johnsonsb"分布的参数
    
    f.hist()  # 样本数据的 normed histogram(面积为1的直方图)
    plt.savefig(path + "f-hist-" + save_path, dpi=300)
    plt.show()
    plt.cla()
    
    f.plot_pdf(Nbest=2, lw=2, method='sumsquare_error')  # Nbest 绘制前几名; lw 宽度; 绘制拟合分布的PDF(概率密度函数)
    plt.savefig(path + "f-nbest-pdf-" + save_path, dpi=300)
    plt.show()
    plt.cla()


def draw_distribution_histogram(nums, save_path, xlabel, ylabel, title, stat="count", bins="auto",
                                binwidth=0.01, binrange=None, is_kde=False, is_cumulative=False):
    """
    https://seaborn.pydata.org/generated/seaborn.histplot.html

    bins: 设置直方图条形的数目，或者是包含一系列值得列表
    binwidth: 每个柱形图的宽度
    binrange: 显示柱形图的范, 如[1.0, 2.0]
    is_kde: 是否绘制核密度图
    is_cumulative : 是否绘累计分布图
    stat：绘制哪种类型的图
        "count": show the number of observations in each bin
        "frequency": show the number of observations divided by the bin width
        "probability": or proportion: normalize such that bar heights sum to 1
        "percent": normalize such that bar heights sum to 100
        "density": normalize such that the total area of the histogram equals 1
    """
    sns.set()  # 切换到sns的默认运行配置
    sns.histplot(data=nums, stat=stat, bins=bins, binwidth=binwidth, binrange=binrange,
                 kde=is_kde, cumulative=is_cumulative, line_kws=dict(), color="steelblue")
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = 'times new roman'
    plt.xlim(xmin=0, xmax=26)
    plt.xticks(range(0, 28, 2))
    
    # 添加x轴和y轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加标题
    plt.title(title)
    plt.tight_layout()  # 处理显示不完整的问题
    # plt.grid()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.cla()


if __name__ == '__main__':

    path = [
        "dist_result/HID_OutdoorGait-euc/dist-2023-0417-145535-train_gaitgl_HID_OutdoorGait_CASIA-B_OUMVLP-test_HID_OutdoorGait-restore_170000-rerank_False-euc.csv",
        "dist_result/HID_OutdoorGait-euc/res-2023-0417-145545-train_gaitgl_HID_OutdoorGait_CASIA-B_OUMVLP-test_HID_OutdoorGait-restore_170000-rerank_False-euc.csv"
    ]
    dataset = "HID_OutdoorGait-euc"
    metric = "euc"
    
    read(path, dataset, metric)
