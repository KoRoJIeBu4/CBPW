import os
import webbrowser
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from IPython.display import display, clear_output

import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import mpld3
import mplcursors

import ipywidgets as widgets
from ipywidgets import interactive, VBox, HBox, interact

from scipy.spatial.distance import cdist
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_reset
import statsmodels.stats.api as sms
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_white,
    het_breuschpagan,
    het_goldfeldquandt,
    het_arch,
)

import pychow
import requests

import streamlit as st


# выкидываем пропуски
def drop_missing_values(df, reference_columns):
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned[reference_columns]

    return df_cleaned


def normalized_dict_by_minmax(dict_of_years: dict[int, pd.DataFrame]):
    neutral = (
        'object_name',
        'object_level',
        'year'
    )

    normilized_dict = {}
    for year in dict_of_years.keys():
        normilized_dict[year] = dict_of_years[year].copy()
        for category in normilized_dict[year].columns:
            if category in neutral:
                continue
            normilized_dict[year][category] = (normilized_dict[year][category] - normilized_dict[year][
                category].min()) / (normilized_dict[year][category].max() - normilized_dict[year][category].min())
    return normilized_dict


def normalized_dict_by_minmax_by_base_year(dict_of_years: dict[int, pd.DataFrame], base_year = 2010):
    min_values_by_features = {}
    max_values_by_features = {}
    neutral = (
        'object_name',
        'object_level',
        'year'
    )

    for category in dict_of_years[base_year].columns:
        if category in neutral:
            continue
        min_values_by_features[category] = dict_of_years[base_year][category].min()
        max_values_by_features[category] = dict_of_years[base_year][category].max()

    normilized_dict = {}
    for year in dict_of_years.keys():
        normilized_dict[year] = dict_of_years[year].copy()
        for category in normilized_dict[year].columns:
            if category in neutral:
                continue
            normilized_dict[year][category] = (normilized_dict[year][category] - min_values_by_features[category]) / (
                        max_values_by_features[category] - min_values_by_features[category])
    return normilized_dict


def calculate_index_with_weights(dict_of_dataframes, way_of_calculating = 1):
    '''
    Вычисляет индекс на основе заданных данных и выбранного метода расчета весов.

    Параметры:
    dict_of_dataframes (dict): Словарь, где ключи - годы, а значения - DataFrame с данными.
    way_of_calculating (int): Метод расчета весов:
        1 - PCA (анализ главных компонент);
        2 - Обратная дисперсия;
        3 - Равные веса.

    Возвращает:
    tuple: Кортеж, содержащий DataFrame с индексами по годам и массив весов.
    '''

    indexed_data = {}
    combined_data = pd.concat(dict_of_dataframes.values())
    numerical_columns = combined_data.drop(columns = ['object_name', 'object_level', 'year'])

    if way_of_calculating == 1:  # PCA
        pca = PCA(n_components = 1)
        pca.fit(numerical_columns)
        weights = np.abs(pca.components_[0])
        weights /= weights.sum()

    elif way_of_calculating == 2:  # Обратная дисперсия
        variances = numerical_columns.var()
        weights = 1 / variances
        weights /= weights.sum()

    else:  # Равные веса
        num_columns = numerical_columns.shape[1]
        weights = np.ones(num_columns) / num_columns

    for year, df in dict_of_dataframes.items():
        numerical_columns = df.drop(columns = ['object_name', 'object_level', 'year'])
        weighted_mean = (numerical_columns * weights).sum(axis = 1)
        df['index'] = weighted_mean
        indexed_data[year] = df[['object_name', 'index']]

    combined_index = pd.DataFrame()
    for year, df in indexed_data.items():
        df = df.set_index('object_name')['index']
        combined_index[year] = df

    return combined_index, weights


def combine_indices(df1, df2, alpha):
    if not df1.shape == df2.shape:
        raise ValueError("Датафреймы должны иметь одинаковую структуру (размеры и индексы).")
    if not (df1.index.equals(df2.index) and df1.columns.equals(df2.columns)):
        raise ValueError("Датафреймы должны иметь одинаковые индексы и столбцы.")

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha должен быть в пределах от 0 до 1.")

    combined_df = alpha * df1 + (1 - alpha) * df2
    return combined_df


def plot_index_trends_multi(dfs_by_base, dfs_by_current, dfs_final, region, domen):
    if not (dfs_by_base.index.equals(dfs_by_current.index) and dfs_by_current.index.equals(dfs_final.index)):
        raise ValueError("Индексы регионов должны совпадать во всех датасетах.")

    region_name = region
    base_index = dfs_by_base.loc[region]
    current_index = dfs_by_current.loc[region]
    final_index = dfs_final.loc[region]
    plt.figure(figsize = (14, 8))
    plt.plot(final_index.index, final_index.values, marker = 'o', linestyle = '-', linewidth = 3,
             label = "Final Index", color = 'red', alpha = 0.9)  # Основной индекс
    plt.plot(base_index.index, base_index.values, marker = 'o', linestyle = '--', linewidth = 2,
             label = "Base Index", color = 'blue', alpha = 0.5)
    plt.plot(current_index.index, current_index.values, marker = 'o', linestyle = ':', linewidth = 2,
             label = "Current Index", color = 'green', alpha = 0.5)
    plt.title(f"{domen}. Динамика индексов по годам для региона: {region_name}", fontsize = 16, fontweight = 'bold')
    plt.xlabel("Год", fontsize = 14)
    plt.ylabel("Индекс", fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.legend(fontsize = 12)
    for i in range(len(final_index)):
        plt.annotate(f"{final_index.values[i]:.2f}",
                     (final_index.index[i], final_index.values[i]),
                     textcoords = "offset points",
                     xytext = (0, 10),
                     ha = 'center', fontsize = 10, color = 'red')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.gca().set_facecolor('#f9f9f9')

    plt.show()


def plot_index_trends_with_similar(dfs_by_base, dfs_by_current, dfs_final, region_name, domen, k = 3):
    if not (dfs_by_base.index.equals(dfs_by_current.index) and dfs_by_current.index.equals(dfs_final.index)):
        raise ValueError("Индексы регионов должны совпадать во всех датасетах.")
    if region_name not in dfs_final.index:
        raise ValueError(f"Регион '{region_name}' отсутствует в данных.")
    target_vector = dfs_final.loc[region_name].values.reshape(1, -1)
    distances = cdist(target_vector, dfs_final.values, metric = 'euclidean').flatten()
    similar_indices = np.argsort(distances)[1:k + 1]
    similar_regions = dfs_final.index[similar_indices]
    plt.figure(figsize = (14, 8))
    plt.gca().set_facecolor('white')
    plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.plot(dfs_final.columns, dfs_final.loc[region_name],
             marker = 'o', linestyle = '-', linewidth = 3, label = f"{region_name} (Target)",
             color = 'orange', alpha = 0.9, zorder = 5)

    colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']

    for i, similar_region in enumerate(similar_regions):
        plt.plot(dfs_final.columns, dfs_final.loc[similar_region],
                 marker = 's', linestyle = ':', linewidth = 2, label = similar_region,
                 color = colors[i % len(colors)], alpha = 0.8)

        max_value = dfs_final.loc[similar_region].max()
        min_value = dfs_final.loc[similar_region].min()
        max_index = dfs_final.loc[similar_region].idxmax()
        min_index = dfs_final.loc[similar_region].idxmin()

        plt.scatter(max_index, max_value, color = colors[i % len(colors)], s = 100, zorder = 10, edgecolor = 'black',
                    marker = 'o')
        plt.scatter(min_index, min_value, color = colors[i % len(colors)], s = 100, zorder = 10, edgecolor = 'black',
                    marker = '^')

    max_value_target = dfs_final.loc[region_name].max()
    min_value_target = dfs_final.loc[region_name].min()
    max_index_target = dfs_final.loc[region_name].idxmax()
    min_index_target = dfs_final.loc[region_name].idxmin()
    plt.scatter(max_index_target, max_value_target, color = 'orange', s = 100, zorder = 10, edgecolor = 'black',
                marker = 'o', label = 'Максимум')
    plt.scatter(min_index_target, min_value_target, color = 'orange', s = 100, zorder = 10, edgecolor = 'black',
                marker = '^', label = 'Минимум')
    plt.title(f"{domen}. Динамика индексов для региона '{region_name}' и {k} похожих регионов", fontsize = 18,
              fontweight = 'bold', color = '#333')
    plt.xlabel("Год", fontsize = 14)
    plt.ylabel("Индекс", fontsize = 14)

    plt.legend(fontsize = 12, loc = "best")
    plt.xticks(rotation = 45)

    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)

    plt.tight_layout()
    plt.show()


def plot_top_successful_regions_dynamic(dfs_final, domen, top_n = 10, alpha = 0.5):
    """
    Строит горизонтальную диаграмму топ-N самых успешных регионов, учитывая уровень индекса за последний год и динамику.
    Добавлены декоративные элементы для улучшения визуализации.

    Args:
        dfs_final (pd.DataFrame): Датафрейм с индексами регионов (индексы - регионы, колонки - годы).
        top_n (int): Количество самых успешных регионов для отображения.
        alpha (float): Вес для динамики (0 <= alpha <= 1).
                       Вес уровня последнего года = 1 - alpha.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Параметр alpha должен быть в диапазоне от 0 до 1.")

    index_growth = dfs_final.diff(axis = 1).mean(axis = 1)
    last_year = dfs_final.columns[-1]
    last_year_index = dfs_final[last_year]
    combined_score = alpha * index_growth + (1 - alpha) * last_year_index
    top_regions = combined_score.sort_values(ascending = False).head(top_n)
    norm = plt.Normalize(min(top_regions), max(top_regions))
    cmap = matplotlib.colormaps['coolwarm']

    fig, ax = plt.subplots(figsize = (12, 7))  # Создаем ось ax
    bars = ax.barh(top_regions.sort_values().index, top_regions.sort_values(),
                   color = cmap(norm(top_regions.sort_values())))

    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        for i, (region, value) in enumerate(zip(top_regions.sort_values().index, top_regions.sort_values())):
            ax.text(value + 0.01, i, f"({value:.3f})", va = 'center', fontsize = 12, color = 'black', weight = 'bold')

    ax.set_title(f"{domen}. Топ-{top_n} самых успешных регионов с учетом динамики ({last_year})", fontsize = 16,
                 weight = 'bold')
    ax.set_xlabel("Куммулятивный рейтинг", fontsize = 14)
    ax.set_ylabel("Регионы", fontsize = 14)
    ax.tick_params(axis = 'x', labelsize = 12)
    ax.tick_params(axis = 'y', labelsize = 12)

    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])

    ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.show()


def plot_bottom_successful_regions_dynamic(dfs_final, domen, bottom_n = 10, alpha = 0.5):
    """
    Строит горизонтальную диаграмму bottom-N наименее успешных регионов, учитывая уровень индекса за последний год и динамику.
    Добавлены декоративные элементы для улучшения визуализации.

    Args:
        dfs_final (pd.DataFrame): Датафрейм с индексами регионов (индексы - регионы, колонки - годы).
        bottom_n (int): Количество наименее успешных регионов для отображения.
        alpha (float): Вес для динамики (0 <= alpha <= 1).
                       Вес уровня последнего года = 1 - alpha.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Параметр alpha должен быть в диапазоне от 0 до 1.")

    index_growth = dfs_final.diff(axis = 1).mean(axis = 1)
    last_year = dfs_final.columns[-1]
    last_year_index = dfs_final[last_year]
    combined_score = alpha * index_growth + (1 - alpha) * last_year_index

    bottom_regions = combined_score.sort_values(ascending = True).head(bottom_n)
    norm = plt.Normalize(min(bottom_regions), max(bottom_regions))
    cmap = matplotlib.colormaps['Reds']  # Используем цветовую схему для негативных результатов

    fig, ax = plt.subplots(figsize = (12, 7))
    bars = ax.barh(bottom_regions.sort_values(ascending = False).index,
                   bottom_regions.sort_values(ascending = False),
                   color = cmap(norm(bottom_regions.sort_values(ascending = False))))

    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        for i, (region, value) in enumerate(zip(bottom_regions.sort_values(ascending = False).index,
                                                bottom_regions.sort_values(ascending = False))):
            ax.text(value + 0.01, i, f"({value:.3f})", va = 'center', fontsize = 12, color = 'black', weight = 'bold')

    ax.set_title(f"{domen}. Топ-{bottom_n} наименее успешных регионов с учетом динамики ({last_year})",
                 fontsize = 16, weight = 'bold', color = 'darkred')
    ax.set_xlabel("Кумулятивный рейтинг", fontsize = 14)
    ax.set_ylabel("Регионы", fontsize = 14)
    ax.tick_params(axis = 'x', labelsize = 12)
    ax.tick_params(axis = 'y', labelsize = 12)

    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])

    ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.show()


def mean_weight(weights_by_base, weights_by_current, alpha = 0.9):
    return weights_by_base * alpha + (1 - alpha) * weights_by_current


def count_impact_per_component(df: pd.DataFrame, weights: np.array):
    '''
    Функция принимает два аргумента:
        df : текущий датафрейм
        weights : веса для признаков
    Return:
        Умножаем веса на значения признаков
    '''
    buff_df = df.copy()
    buff_df.iloc[:, 3:-1] = buff_df.iloc[:, 3:-1] * weights
    return buff_df


def combine_for_weights_and_importances(dfs1, dfs2, index_share):
    '''
    Функция принимает два датафрейма и коэффициент сглаживания
    Нужна, чтобы вычислять важность признака в структуре финального индекса
    Возвращает важность каждого признака в долях
    '''
    dfs_final = {}
    for year in dfs1.keys():
        dfs_final[year] = dfs1[year].copy()
        dfs_final[year].iloc[:, 3:-1] = dfs_final[year].iloc[:, 3:-1] * index_share + dfs2[year].iloc[:, 3:-1] * (
                    1 - index_share)
        total_sum = dfs_final[year].iloc[:, 3:-1].sum(axis = 1)
        dfs_final[year] = dfs_final[year].drop('index', axis = 1)
        for col in dfs_final[year].columns[3:]:
            dfs_final[year][col] /= total_sum
    return dfs_final


def rank_importances(clear_domen, domen, importances_of_domen, region, year):
    if isinstance(year, list):
        start = year[0]
        end = year[1]
        df = importances_of_domen[start][importances_of_domen[start]['object_name'] == region].iloc[:, 3:].transpose()
        df.columns = ['Влияние на ИРР']
        for year in range(start + 1, end + 1, 1):
            chosen = importances_of_domen[year][importances_of_domen[year]['object_name'] == region].iloc[:,
                     3:].transpose()
            chosen.columns = ['Влияние на ИРР']
            df += chosen
        df['Влияние на ИРР'] = df['Влияние на ИРР'] / (end - start + 1)
        df = df.sort_values(by = 'Влияние на ИРР', ascending = False)
        return df
    else:
        chosen = importances_of_domen[year][importances_of_domen[year]['object_name'] == region].iloc[:, 3:].transpose()
        chosen.columns = ['Влияние на ИРР']
        values = clear_domen[year][clear_domen[year]['object_name'] == region].iloc[:, 3:].transpose()
        values.columns = ['Значение показателя']
        chosen = pd.concat([chosen, values], axis = 1)
        chosen = chosen.sort_values(by = 'Влияние на ИРР', ascending = False)

        category = {'Положительно': [], 'Отрицательно': []}
        clear = clear_domen[year][clear_domen[year]['object_name'] == region].iloc[:, 3:]
        modified = domen[year][domen[year]['object_name'] == region].iloc[:, 3:]

        for col in clear.columns:
            if clear[col].values == modified[col].values:
                category['Положительно'].append(col)
            else:
                category['Отрицательно'].append(col)
        chosen['Тип показателя'] = chosen.index.to_series().apply(
            lambda x: 'Стимулянт' if x in category['Положительно'] else 'Дестимулянт'
        )

        # гарантия нормировки
        total = chosen['Влияние на ИРР'].sum()
        chosen['Влияние на ИРР'] = np.abs(chosen['Влияние на ИРР']) / total
        chosen = chosen.sort_values(by = 'Влияние на ИРР', ascending = False)

        return chosen


def ravel_domen_final(data):
    '''
    data: pd.DataFrame
    Вытягиевает в формате столбцов: год, регион, значение индекса
    '''
    data = data.reset_index()
    melted_data = data.melt(id_vars = [data.columns[0]],
                            var_name = 'год',
                            value_name = 'значение индекса')

    melted_data.rename(columns = {data.columns[0]: 'регион'}, inplace = True)

    return melted_data


def ravel_domen_dict(data_dict):
    '''
    data_dict: dict
    Словарь, где ключи - годы, значения - DataFrame с колонками: object_name, object_level, year и другие признаки.

    Возвращает объединенный DataFrame с колонками: год, object_name, object_level и остальные признаки.
    '''
    df_list = []

    for year, df in data_dict.items():
        if df.empty:
            raise ValueError(f"DataFrame for year {year} is empty.")
        if not all(col in df.columns for col in ['object_name', 'object_level', 'year']):
            raise ValueError(f"DataFrame for year {year} is missing required columns.")

        df['year'] = year

        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index = True)

    return combined_df


def analyze_indicator(df, agg_func = "mean", region = None):
    """
    Интерактивно анализирует динамику показателя по годам, с возможностью фильтрации по региону.

    :param df: Dict[int : pd.DataFrame]
    :param agg_func: "mean" (по умолчанию) или "median" - метод агрегирования
    :param region: Название региона (или None, чтобы смотреть по всем)
    """
    # 1️⃣ Фильтрация по региону, если указан
    df = ravel_domen_dict(df)
    if region:
        df = df[df["object_name"] == region]

    # 2️⃣ Выбор показателя
    indicators = df.columns[3:]  # Все столбцы после 'year'
    print("\nВыберите показатель для анализа:")
    for i, col in enumerate(indicators, 1):
        print(f"{i}) {col}")

    choice = int(input("\nВведите номер показателя: ")) - 1
    indicator = indicators[choice]

    # 3️⃣ Агрегация данных (по годам)
    if agg_func == "mean":
        df_grouped = df.groupby("year")[indicator].mean()
    elif agg_func == "median":
        df_grouped = df.groupby("year")[indicator].median()
    else:
        raise ValueError("Неверное значение agg_func. Используйте 'mean' или 'median'.")

    # 4️⃣ Расчет стандартного отклонения для доверительного интервала
    df_std = df.groupby("year")[indicator].std()

    # 5️⃣ Автоматический тренд (линейная регрессия)
    years = df_grouped.index.values
    y_values = df_grouped.values

    # Выполнение линейной регрессии
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, y_values)

    # Проверка значимости коэффициента наклона
    if p_value < 0.05:  # Уровень значимости 0.05
        trend_slope = slope
        if trend_slope > 0:
            trend = "Восходящий 📈"
        elif trend_slope < 0:
            trend = "Нисходящий 📉"
    else:
        trend = "Стационарный ➖"

    region_str = f" ({region})" if region else " (по всем регионам)"
    print(f"\n📊 Тренд показателя '{indicator}'{region_str}: {trend}")

    # 6️⃣ Темпы роста (% изменения от прошлого года)
    df_pct_change = df_grouped.pct_change() * 100

    # 7️⃣ Топ-3 года по значению
    top_years = df_grouped.nlargest(3)
    print("\n🏆 Топ-3 года по показателю:")
    print(top_years)

    # 🔹 Визуализация 🔹
    fig, axes = plt.subplots(2, 1, figsize = (12, 10))

    # --- Первый график: Динамика показателя ---
    axes[0].plot(df_grouped.index, df_grouped.values, marker = "o", label = "Динамика показателя", color = "b")
    axes[0].plot(years, intercept + slope * years, linestyle = "dashed", color = "red", label = "Линия тренда")
    axes[0].fill_between(df_grouped.index, df_grouped - df_std, df_grouped + df_std, color = "gray", alpha = 0.2)

    axes[0].set_xlabel("Год")
    # axes[0].set_title(f"Динамика показателя: {indicator}{region_str}")
    axes[0].legend()
    axes[0].grid(True)

    # --- Второй график: Темпы роста (столбиками) ---
    colors = ["green" if val >= 0 else "red" for val in df_pct_change.values]

    bars = axes[1].bar(df_pct_change.index, df_pct_change.values, color = colors, alpha = 0.7)

    # 🔢 Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):  # Проверка, чтобы не писать NaN
            axes[1].text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%",
                         ha = "center", va = "bottom" if height > 0 else "top", fontsize = 10, color = "black")

    axes[1].set_xlabel("Год")
    axes[1].set_ylabel("Темпы прироста (%)")
    axes[1].set_title(f"Темпы прироста показателя по годам{region_str}")
    axes[1].axhline(0, color = "gray", linestyle = "dashed")  # Горизонтальная линия на 0%
    axes[1].grid(True)

    plt.tight_layout()


def analyze_indicator_interactive(df):
    """
    Интерактивная версия функции analyze_indicator с выбором региона и показателя.
    """
    df = ravel_domen_dict(df)  # Преобразование данных

    # Виджет для выбора региона
    region_widget = widgets.Dropdown(
        options = [None] + list(df["object_name"].unique()),  # None = все регионы
        value = None,
        description = "Регион:"
    )

    # Виджет для выбора метода агрегирования
    agg_widget = widgets.RadioButtons(
        options = ["mean", "median"],
        value = "mean",
        description = "Агрегация:"
    )

    # Виджет для выбора показателя (обновляется динамически)
    indicator_widget = widgets.Dropdown(
        options = df.columns[3:],
        description = "Показатель:"
    )

    # Функция обработки
    def process(region, agg_func, indicator):
        df_filtered = df[df["object_name"] == region] if region else df

        # Агрегация данных
        if agg_func == "mean":
            df_grouped = df_filtered.groupby("year")[indicator].mean()
        else:
            df_grouped = df_filtered.groupby("year")[indicator].median()

        # Стандартное отклонение
        df_std = df_filtered.groupby("year")[indicator].std()

        # Линейная регрессия
        years = df_grouped.index.values
        y_values = df_grouped.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, y_values)

        trend = "Стационарный ➖"
        if p_value < 0.05:
            if slope > 0:
                trend = "Восходящий 📈"
            elif slope < 0:
                trend = "Нисходящий 📉"

        print(f"\n📊 Тренд показателя '{indicator}' ({region if region else 'по всем регионам'}): {trend}")

        # Темпы роста
        df_pct_change = df_grouped.pct_change() * 100

        # Визуализация
        fig, axes = plt.subplots(2, 1, figsize = (12, 10))

        # --- График динамики ---
        axes[0].plot(df_grouped.index, df_grouped.values, marker = "o", label = "Динамика", color = "b")
        axes[0].plot(years, intercept + slope * years, linestyle = "dashed", color = "red", label = "Тренд")
        axes[0].fill_between(df_grouped.index, df_grouped - df_std, df_grouped + df_std, color = "gray", alpha = 0.2)
        axes[0].legend()
        axes[0].grid(True)

        # --- График темпов роста ---
        colors = ["green" if val >= 0 else "red" for val in df_pct_change.values]
        bars = axes[1].bar(df_pct_change.index, df_pct_change.values, color = colors, alpha = 0.7)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                axes[1].text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%",
                             ha = "center", va = "bottom" if height > 0 else "top", fontsize = 10)

        axes[1].axhline(0, color = "gray", linestyle = "dashed")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    # Создание интерактивного интерфейса
    interactive_plot = interactive(process,
                                   region = region_widget,
                                   agg_func = agg_widget,
                                   indicator = indicator_widget)

    display(interactive_plot)


def save_hdf5(data: dict[int, pd.DataFrame], filename: str) -> None:
    """
    Сохраняет словарь DataFrame в HDF5 файл.

    :param data: Словарь, где ключи - годы (int), а значения - DataFrame.
    :param filename: Имя файла для сохранения данных.
    """
    with pd.HDFStore(filename, mode = "w") as store:
        for year, df in data.items():
            store.put(f"year_{year}", df)


def load_hdf5(filename: str) -> dict[int, pd.DataFrame]:
    """
    Загружает данные из HDF5 файла в словарь DataFrame.

    :param filename: Имя файла для загрузки данных.
    :return: Словарь, где ключи - годы (int), а значения - DataFrame.
    """
    data = {}
    with pd.HDFStore(filename, mode = "r") as store:
        for key in store.keys():
            year = int(key.split("_")[1])  # Извлекаем год из ключа
            data[year] = store[key]

    return data

def remove_highly_correlated_features(df: pd.DataFrame, correlation_threshold=0.9) -> pd.DataFrame:
    """
    Удаляет признаки с высокой корреляцией из датафрейма.
    
    :param df: pandas DataFrame, исходный датафрейм
    :param correlation_threshold: float, порог корреляции для удаления признаков
    :return: pandas DataFrame, датафрейм без высококоррелирующих признаков
    """
    
    # Фильтруем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Вычисляем матрицу корреляции для числовых столбцов
    corr_matrix = numeric_df.corr().abs()
    
    # Создаем маску для верхнего треугольника матрицы
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Находим признаки с высокой корреляцией
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    # Удаляем высококоррелирующие признаки из оригинального датафрейма
    df_reduced = df.drop(columns=to_drop)
        
    return df_reduced

def process_dataframes_to_reduce_multicollinearity(dataframes: dict[int, pd.DataFrame], correlation_threshold=0.9) -> dict[int, pd.DataFrame]:
    """
    Обрабатывает словарь датафреймов, удаляя высококоррелирующие признаки из каждого.
    
    :param dataframes: Dict[int, pd.DataFrame], словарь с датафреймами
    :param correlation_threshold: float, порог корреляции для удаления признаков
    :return: Dict[int, pd.DataFrame], словарь с обработанными датафреймами
    """
    
    # Собираем все датафреймы в один для анализа корреляции
    combined_df = pd.concat(dataframes.values(), axis=0)
    
    # Определяем признаки для удаления на основе объединенного датафрейма
    numeric_combined_df = combined_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_combined_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        
    # Удаляем эти признаки из каждого датафрейма
    processed_dataframes = {key: df.drop(columns=to_drop) for key, df in dataframes.items()}
    
    return processed_dataframes

def to_one_structure(decomposed_SIIRD):
    """
    Приводит словарь типа {std : pd.DataFrame} к словарю с одинаковыми индексами и шейпами.
    Все DataFrame отсортированы по индексам и столбцам.

    Parameters:
    - decomposed_SIIRD (dict): Словарь, где ключи — это стандарты, а значения — DataFrame.

    Returns:
    - dict: Словарь, где все DataFrame имеют одинаковые индексы и формы, отсортированные по индексам и столбцам.
    """
    # Определяем общие индексы и столбцы для всех DataFrame
    common_index = None
    common_columns = None

    # Находим пересечение индексов и столбцов
    for key, df in decomposed_SIIRD.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

        if common_columns is None:
            common_columns = df.columns
        else:
            common_columns = common_columns.intersection(df.columns)

    # Сортируем индексы и столбцы
    common_index = sorted(common_index)
    common_columns = sorted(common_columns)

    # Приводим все DataFrame к одинаковым индексам и столбцам
    for key, df in decomposed_SIIRD.items():
        # Обрезаем DataFrame по общим индексам и столбцам и сортируем
        decomposed_SIIRD[key] = df.loc[common_index, common_columns]
    
    return decomposed_SIIRD

def norm_final_domen_by_minmax_normalizing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует переданные DataFrame-матрицы по min-max и агрегирует их в один.
    Одновременная нормализация вдоль времени и вдоль признаков оправда тем, что мы фиксириуем заданный отрезок времени и изучаем
    состояние в таком стационаре. То есть набор [data] ограничивает не только рассматриваемый промежуток, но и показатели. Он постулирует, что в рамках
    предложенных данных есть лучшее (это 1) и худшее (это 0) положение дел.

    :param data: DataFrame (матрица регионов × лет)
    :return: нормированный DataFrame
    """
    df = data.copy()
    x_max = df.max().max()
    x_min = df.min().min()
    return (df - x_min) / (x_max - x_min)


def compute_weights_for_SIIRD(decomposed_SIIRD, method='pca'):
    """
    Вычисляет веса для доменов из decomposed_SIIRD с использованием нескольких методов.
    Каждый DataFrame из decomposed_SIIRD имеет индексы: регионы, колонки: года.
    
    Для каждого домена:
      1. Заполняем пропуски средним значением по столбцу (предполагается, что DataFrame уже очищен).
      2. «Растягиваем» (flatten) матрицу в вектор.
    
    Затем вычисляем веса по методам:
      - 'equal': равные веса.
      - 'variance': веса пропорциональны дисперсии.
      - 'inverse_variance': веса, обратные дисперсии.
      - 'pca': веса, полученные на основе первой главной компоненты.
    
    Возвращает:
      dict: словарь, где ключи – имена методов, а значения – словари весов для каждого домена.
    """
    domains = list(decomposed_SIIRD.keys())
    vectors = {}
    for domain in domains:
        df = decomposed_SIIRD[domain].copy()
        # Здесь предполагаем, что пропуски уже заполнены (если нет, можно добавить fillna здесь)
        vectors[domain] = df.values.flatten()
    
    # Метод 1: Равные веса
    equal_weights = {domain: 1 / len(domains) for domain in domains}
    
    # Метод 2: Веса пропорциональны дисперсии
    var_dict = {domain: np.var(vectors[domain]) for domain in domains}
    total_var = sum(var_dict.values())
    variance_weights = {domain: var_dict[domain] / total_var if total_var != 0 else 1/len(domains)
                        for domain in domains}
    
    # Метод 3: Веса обратной дисперсии
    inv_var_dict = {domain: 1/np.var(vectors[domain]) if np.var(vectors[domain]) != 0 else 0
                    for domain in domains}
    total_inv = sum(inv_var_dict.values())
    inverse_variance_weights = {domain: inv_var_dict[domain] / total_inv if total_inv != 0 else 1/len(domains)
                                for domain in domains}
    
    # Метод 4: PCA
    X = np.column_stack([vectors[domain] for domain in domains])
    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = np.abs(pca.components_[0])
    loadings = loadings / loadings.sum()
    pca_weights = {domain: loadings[i] for i, domain in enumerate(domains)}
    
    # Выбираем нужный метод
    if method == 'equal':
        return equal_weights
    elif method == 'variance':
        return variance_weights
    elif method == 'inverse_variance':
        return inverse_variance_weights
    elif method == 'pca':
        return pca_weights
    else:
        raise ValueError("Неизвестный метод. Используйте 'equal', 'variance', 'inverse_variance' или 'pca'.")

def compose_SIIRD(decomposed_SIIRD, weight_method='pca'):
    """
    Формирует интегральный социальный индекс SIIRD.
    
    1. Приводит все DataFrame в decomposed_SIIRD к одной структуре (одинаковые регионы и годы).
    2. Заполняет пропуски средним значением по столбцу.
    3. Вычисляет веса для каждого домена по выбранному методу (по умолчанию PCA).
    4. Складывает все матрицы с учетом весов.
    5. Нормализует итоговую матрицу SIIRD с помощью функции norm_final_domen_by_minmax_normalizing.
    
    Returns:
      pd.DataFrame: Итоговый SIIRD.
    """
    # 1. Приводим к общей структуре и сортируем
    decomposed_SIIRD = to_one_structure(decomposed_SIIRD)
    
    # 2. Заполняем пропуски средним значением по столбцу для каждого домена
    for domain in decomposed_SIIRD:
        df = decomposed_SIIRD[domain]
        decomposed_SIIRD[domain] = df.apply(lambda col: col.fillna(col.mean()), axis=0)
    
    # 3. Вычисляем веса для доменов по выбранному методу
    weights = compute_weights_for_SIIRD(decomposed_SIIRD, method=weight_method)
    
    # 4. Складываем все матрицы, умножая каждую на соответствующий вес
    SIIRD = None
    for domain, df in decomposed_SIIRD.items():
        weighted_df = df * weights[domain]
        if SIIRD is None:
            SIIRD = weighted_df.copy()
        else:
            SIIRD += weighted_df
    
    # 5. Нормализуем итоговую матрицу SIIRD с помощью функции norm_final_domen_by_minmax_normalizing
    SIIRD_norm = norm_final_domen_by_minmax_normalizing(SIIRD)
    
    return SIIRD_norm

def compose_SFIIRD(SFIIRD, alpha=0.5):
    """
    Вычисляет итоговый индекс как взвешенную сумму FIIRD и SIIRD.
    
    Параметры:
      SFIIRD: dict, содержащий два DataFrame:
             "FIIRD" – финансовый индекс (регионы x годы)
             "SIIRD" – социальный индекс (регионы x годы)
      alpha: вес для FIIRD, а (1 - alpha) – вес для SIIRD.
    
    Возвращает:
      DataFrame с итоговым индексом (те же регионы и годы).
    """
    FIIRD = SFIIRD["FIIRD"]
    SIIRD = SFIIRD["SIIRD"]
    
    # Проверка, что индексы и колонки совпадают (при необходимости можно привести к одной структуре)
    if not (FIIRD.index.equals(SIIRD.index) and FIIRD.columns.equals(SIIRD.columns)):
        raise ValueError("Индексы и/или столбцы FIIRD и SIIRD не совпадают.")
    
    final_index = alpha * FIIRD + (1 - alpha) * SIIRD
    return final_index

