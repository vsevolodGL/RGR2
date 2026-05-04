import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHA = 0.05
# В папке нет отдельной карточки варианта; если преподавательская карточка
# задает другие параметры для D-1, достаточно поменять эти две константы.
X3_MU0 = 100.0
X4_LAMBDA = 0.2
DATA_FILE = "RGR2_D-1_X1-X4.csv"
PLOTS_DIR = Path("plots")


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def betacf(a, b, x):
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def student_t_cdf(t, df):
    x = df / (df + t * t)
    ib = regularized_beta(x, df / 2.0, 0.5)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_ppf(p, df):
    lo, hi = -100.0, 100.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def regularized_gamma_p(a, x):
    if x <= 0:
        return 0.0
    eps = 1e-14
    fpmin = 1e-300
    gln = math.lgamma(a)

    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                return total * math.exp(-x + a * math.log(x) - gln)
        return total * math.exp(-x + a * math.log(x) - gln)

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return 1.0 - q


def chi2_cdf(x, df):
    return regularized_gamma_p(df / 2.0, x / 2.0)


def chi2_ppf(p, df):
    lo, hi = 0.0, max(1.0, df)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fmt(x, digits=5):
    if isinstance(x, str):
        return x
    if math.isinf(x):
        return "+∞"
    return f"{x:.{digits}f}"


def describe(series):
    x = np.asarray(series, dtype=float)
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "var_biased": float(np.var(x, ddof=0)),
        "var_unbiased": float(np.var(x, ddof=1)),
        "std_unbiased": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q1": float(np.quantile(x, 0.25)),
        "q3": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def pooled_t_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    stat = (mean1 - mean2) / math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    p_value = 2.0 * (1.0 - student_t_cdf(abs(stat), df))
    crit = student_t_ppf(1.0 - ALPHA / 2.0, df)
    return {
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "var1": var1,
        "var2": var2,
        "sp2": sp2,
        "df": df,
        "stat": stat,
        "p_value": p_value,
        "crit": crit,
    }


def one_sample_t_test(x, mu0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    df = n - 1
    stat = (mean - mu0) / (std / math.sqrt(n))
    p_value = 2.0 * (1.0 - student_t_cdf(abs(stat), df))
    crit = student_t_ppf(1.0 - ALPHA / 2.0, df)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "df": df,
        "stat": stat,
        "p_value": p_value,
        "crit": crit,
    }


def mann_whitney_u_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.concatenate([x, y])
    groups = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    n1, n2 = len(x), len(y)
    r1 = float(ranks[groups == 0].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2.0

    _, counts = np.unique(values, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    n = n1 + n2
    var_u = n1 * n2 / 12.0 * ((n + 1) - tie_sum / (n * (n - 1)))
    z = (u - mean_u + 0.5) / math.sqrt(var_u)
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return {
        "n1": n1,
        "n2": n2,
        "r1": r1,
        "u1": u1,
        "u2": u2,
        "u": u,
        "mean_u": mean_u,
        "var_u": var_u,
        "z": z,
        "p_value": p_value,
    }


def pearson_exp_test(x, lam, bins_count=6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    edges = []
    for p in np.linspace(0.0, 1.0, bins_count + 1):
        if p == 0.0:
            edges.append(0.0)
        elif p == 1.0:
            edges.append(math.inf)
        else:
            edges.append(-math.log(1.0 - p) / lam)

    observed = []
    for left, right in zip(edges[:-1], edges[1:]):
        if math.isinf(right):
            count = int(np.sum(x >= left))
        else:
            count = int(np.sum((x >= left) & (x < right)))
        observed.append(count)

    probabilities = [1.0 / bins_count] * bins_count
    expected = [n / bins_count] * bins_count
    contributions = [(o - e) ** 2 / e for o, e in zip(observed, expected)]
    stat = float(sum(contributions))
    df = bins_count - 1
    p_value = 1.0 - chi2_cdf(stat, df)
    crit = chi2_ppf(1.0 - ALPHA, df)
    return {
        "edges": edges,
        "observed": observed,
        "probabilities": probabilities,
        "expected": expected,
        "contributions": contributions,
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "crit": crit,
    }


def make_plots(df, desc):
    PLOTS_DIR.mkdir(exist_ok=True)
    x1 = df["X1"].to_numpy()
    x2 = df["X2"].to_numpy()
    x3 = df["X3"].to_numpy()
    x4 = df["X4"].to_numpy()

    plt.figure(figsize=(8, 4.8))
    bins = np.histogram_bin_edges(np.concatenate([x1, x2]), bins="scott")
    plt.hist(x1, bins=bins, alpha=0.58, density=True, edgecolor="black", label="X1")
    plt.hist(x2, bins=bins, alpha=0.48, density=True, edgecolor="black", label="X2")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.title("Сравнение распределений X1 и X2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "x1_x2_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 4.8))
    plt.boxplot([x1, x2], tick_labels=["X1", "X2"], showmeans=True)
    plt.ylabel("Значение")
    plt.title("Boxplot для X1 и X2")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "x1_x2_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    bins = np.histogram_bin_edges(x3, bins="scott")
    plt.hist(x3, bins=bins, density=True, alpha=0.65, edgecolor="black", label="X3")
    grid = np.linspace(min(x3) - 5, max(x3) + 5, 400)
    s = desc["X3"]["std_unbiased"]
    normal_pdf = np.exp(-0.5 * ((grid - X3_MU0) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    plt.plot(grid, normal_pdf, color="#b3261e", linewidth=2, label=f"N({X3_MU0:.0f}, s^2)")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.title("Гистограмма X3 и нормальная модель")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "x3_hist_normal.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    bins = np.histogram_bin_edges(x4, bins="scott")
    plt.hist(x4, bins=bins, density=True, alpha=0.65, edgecolor="black", label="X4")
    grid = np.linspace(0, max(x4) + 2, 400)
    exp_pdf = X4_LAMBDA * np.exp(-X4_LAMBDA * grid)
    plt.plot(grid, exp_pdf, color="#0b6b58", linewidth=2, label=f"Exp(lambda={X4_LAMBDA:.1f})")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.title("Гистограмма X4 и экспоненциальная модель")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "x4_hist_exp.png", dpi=160)
    plt.close()


def interval_label(left, right):
    if math.isinf(right):
        return f"[{fmt(left, 3)}; +∞)"
    return f"[{fmt(left, 3)}; {fmt(right, 3)})"


def markdown_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report(df, desc, pooled, x3_test, mw, pearson):
    pearson_rows = []
    for left, right, obs, prob, exp, contrib in zip(
        pearson["edges"][:-1],
        pearson["edges"][1:],
        pearson["observed"],
        pearson["probabilities"],
        pearson["expected"],
        pearson["contributions"],
    ):
        pearson_rows.append(
            [
                interval_label(left, right),
                str(obs),
                fmt(prob, 5),
                fmt(exp, 3),
                fmt(contrib, 5),
            ]
        )

    total_obs = sum(pearson["observed"])
    pearson_rows.append(["Итого", str(total_obs), "1.00000", fmt(float(total_obs), 3), fmt(pearson["stat"])])

    return f"""# Расчётно-графическая работа №2

## Вариант D-1

---

## Исходные данные

Используется файл `RGR2_D-1_X1-X4.csv`.

```math
n={len(df)}, \\qquad \\alpha={ALPHA}
```

| Характеристика | X1 | X2 | X3 | X4 |
|---|---:|---:|---:|---:|
| Выборочное среднее | {fmt(desc['X1']['mean'])} | {fmt(desc['X2']['mean'])} | {fmt(desc['X3']['mean'])} | {fmt(desc['X4']['mean'])} |
| Несмещённая дисперсия | {fmt(desc['X1']['var_unbiased'])} | {fmt(desc['X2']['var_unbiased'])} | {fmt(desc['X3']['var_unbiased'])} | {fmt(desc['X4']['var_unbiased'])} |
| Несмещённое стандартное отклонение | {fmt(desc['X1']['std_unbiased'])} | {fmt(desc['X2']['std_unbiased'])} | {fmt(desc['X3']['std_unbiased'])} | {fmt(desc['X4']['std_unbiased'])} |
| Медиана | {fmt(desc['X1']['median'])} | {fmt(desc['X2']['median'])} | {fmt(desc['X3']['median'])} | {fmt(desc['X4']['median'])} |
| Минимум | {fmt(desc['X1']['min'])} | {fmt(desc['X2']['min'])} | {fmt(desc['X3']['min'])} | {fmt(desc['X4']['min'])} |
| Максимум | {fmt(desc['X1']['max'])} | {fmt(desc['X2']['max'])} | {fmt(desc['X3']['max'])} | {fmt(desc['X4']['max'])} |

![x1_x2_hist](plots/x1_x2_hist.png)

![x1_x2_boxplot](plots/x1_x2_boxplot.png)

---

## 4.1. Постановка гипотез и схема проверки

Для пунктов 4.2-4.5 используется единая схема проверки статистической гипотезы:

1. формулируются нулевая и альтернативная гипотезы;
2. задаётся уровень значимости;
3. выбирается статистический критерий и его статистика;
4. вычисляется наблюдаемое значение статистики и p-value или критическая область;
5. принимается решение: если p-value меньше уровня значимости, гипотеза H<sub>0</sub> отвергается.

Уровень значимости во всех проверках:

```math
\\alpha={ALPHA}.
```

В работе используется двусторонняя альтернатива, если не указано иное.

| Пункт | Нулевая гипотеза | Альтернатива | Критерий | Ошибка первого рода |
|---|---|---|---|---|
| 4.2 | E[X1] = E[X2] | E[X1] != E[X2] | двухвыборочный t-критерий Стьюдента | признать средние различными, хотя они равны |
| 4.3 | mu3 = {fmt(X3_MU0, 1)} | mu3 != {fmt(X3_MU0, 1)} | одновыборочный t-критерий | отвергнуть верное значение математического ожидания |
| 4.4 | распределения X1 и X2 одинаковы | распределения X1 и X2 различаются | критерий Манна-Уитни | признать распределения различными, хотя они одинаковы |
| 4.5 | X4 имеет распределение Exp(lambda={fmt(X4_LAMBDA, 1)}) | X4 не имеет это распределение | критерий согласия Пирсона | отвергнуть верную модель распределения |

---

## 4.2. Проверка гипотезы о равенстве математических ожиданий для двух независимых выборок

По столбцам X<sub>1</sub> и X<sub>2</sub> проверяется гипотеза о равенстве математических ожиданий.

Гистограммы и boxplot показывают сдвиг X<sub>2</sub> вправо относительно X<sub>1</sub>, а выборочные средние отличаются на

```math
\\bar X_1-\\bar X_2={fmt(pooled['mean1']-pooled['mean2'])}.
```

### Гипотезы

```math
H_0:E[X_1]=E[X_2],
```

```math
H_1:E[X_1]\\ne E[X_2].
```

Уровень значимости:

```math
\\alpha={ALPHA}.
```

### Выбор критерия

Используем двухвыборочный t-критерий Стьюдента для независимых выборок с неизвестными, но предполагаемо равными дисперсиями:

```math
T=
\\frac{{\\bar X_1-\\bar X_2}}
{{S_p\\sqrt{{\\frac1{{n_1}}+\\frac1{{n_2}}}}}},
```

```math
S_p^2=
\\frac{{(n_1-1)S_1^2+(n_2-1)S_2^2}}
{{n_1+n_2-2}}.
```

При H<sub>0</sub>:

```math
T\\sim t_{{n_1+n_2-2}}.
```

Ошибка первого рода в этой задаче: отвергнуть равенство математических ожиданий X<sub>1</sub> и X<sub>2</sub>, хотя в действительности их математические ожидания равны.

### Вычисления

```math
n_1=n_2={pooled['n1']},
```

```math
\\bar X_1={fmt(pooled['mean1'])},\\qquad \\bar X_2={fmt(pooled['mean2'])}.
```

```math
S_1^2={fmt(pooled['var1'])},\\qquad S_2^2={fmt(pooled['var2'])}.
```

```math
S_p^2={fmt(pooled['sp2'])}.
```

Наблюдаемое значение статистики:

```math
T_{{набл}}={fmt(pooled['stat'])}.
```

Число степеней свободы:

```math
\\nu=n_1+n_2-2={pooled['df']}.
```

Критическое значение для двусторонней альтернативы:

```math
t_{{{pooled['df']};0.975}}={fmt(pooled['crit'])}.
```

Критическая область:

```math
W=(-\\infty;-{fmt(pooled['crit'])}]\\cup[{fmt(pooled['crit'])};+\\infty).
```

p-value:

```math
p={fmt(pooled['p_value'])}.
```

### Вывод

```math
|T_{{набл}}|={fmt(abs(pooled['stat']))}>{fmt(pooled['crit'])}, \\qquad p={fmt(pooled['p_value'])}<0.05.
```

На уровне значимости 0.05 гипотеза H<sub>0</sub> отвергается. Есть статистически значимые основания считать, что математические ожидания X<sub>1</sub> и X<sub>2</sub> различны. По выборке среднее X<sub>2</sub> выше среднего X<sub>1</sub>.

---

## 4.3. Проверка гипотезы о параметре нормального распределения

По столбцу X<sub>3</sub> проверяется гипотеза о математическом ожидании нормального распределения.

![x3_hist_normal](plots/x3_hist_normal.png)

### Гипотезы

```math
H_0:\\mu_3={fmt(X3_MU0, 1)},
```

```math
H_1:\\mu_3\\ne {fmt(X3_MU0, 1)}.
```

Уровень значимости:

```math
\\alpha={ALPHA}.
```

### Выбор критерия

Для нормальной выборки с неизвестной дисперсией используем одновыборочный t-критерий:

```math
T=\\frac{{\\sqrt n(\\bar X_3-\\mu_0)}}{{S_3}}.
```

При H<sub>0</sub>:

```math
T\\sim t_{{n-1}}.
```

Ошибка первого рода: отвергнуть утверждение о том, что математическое ожидание X<sub>3</sub> равно {fmt(X3_MU0, 1)}, хотя оно действительно равно этому значению.

### Вычисления

```math
n={x3_test['n']},\\qquad \\bar X_3={fmt(x3_test['mean'])},\\qquad S_3={fmt(x3_test['std'])}.
```

```math
T_{{набл}}=
\\frac{{\\sqrt{{{x3_test['n']}}}({fmt(x3_test['mean'])}-{fmt(X3_MU0, 1)})}}{{{fmt(x3_test['std'])}}}
={fmt(x3_test['stat'])}.
```

```math
\\nu=n-1={x3_test['df']},\\qquad t_{{{x3_test['df']};0.975}}={fmt(x3_test['crit'])}.
```

p-value:

```math
p={fmt(x3_test['p_value'])}.
```

### Вывод

```math
|T_{{набл}}|={fmt(abs(x3_test['stat']))}>{fmt(x3_test['crit'])}, \\qquad p={fmt(x3_test['p_value'])}<0.05.
```

На уровне значимости 0.05 гипотеза H<sub>0</sub> отвергается. Наблюдаемая выборка даёт статистически значимые основания считать, что математическое ожидание X<sub>3</sub> отличается от {fmt(X3_MU0, 1)}.

---

## 4.4. Непараметрический критерий для двух выборок

Для тех же выборок X<sub>1</sub> и X<sub>2</sub> применим непараметрический критерий сравнения двух независимых выборок.

### Гипотезы

```math
H_0:F_{{X_1}}(x)=F_{{X_2}}(x),
```

```math
H_1:F_{{X_1}}(x)\\ne F_{{X_2}}(x).
```

Уровень значимости:

```math
\\alpha={ALPHA}.
```

### Выбор критерия

Используем критерий Манна-Уитни для двух независимых выборок. Он не требует предположения о нормальности и сравнивает положения двух распределений через ранги объединённой выборки.

Ошибка первого рода: отвергнуть однородность распределений X<sub>1</sub> и X<sub>2</sub>, когда распределения в действительности одинаковы.

### Вычисления

```math
n_1=n_2={mw['n1']}.
```

Сумма рангов первой выборки:

```math
R_1={fmt(mw['r1'])}.
```

Статистики:

```math
U_1={fmt(mw['u1'])},\\qquad U_2={fmt(mw['u2'])}.
```

```math
U=\\min(U_1,U_2)={fmt(mw['u'])}.
```

Для больших выборок используем нормальную аппроксимацию:

```math
E[U]={fmt(mw['mean_u'])},\\qquad D[U]={fmt(mw['var_u'])}.
```

```math
Z_{{набл}}={fmt(mw['z'])}.
```

p-value:

```math
p={fmt(mw['p_value'])}.
```

### Вывод и сравнение с параметрическим критерием

```math
p={fmt(mw['p_value'])}<0.05.
```

Гипотеза H<sub>0</sub> отвергается. Непараметрический критерий также показывает статистически значимое различие между X<sub>1</sub> и X<sub>2</sub>. Вывод совпадает с результатом t-критерия: X<sub>2</sub> имеет более высокое положение распределения, что видно и по среднему, и по рангам.

---

## 4.5. Критерий согласия

По столбцу X<sub>4</sub> проверяется гипотеза согласия с законом распределения, указанным для варианта: экспоненциальное распределение с параметром \\(\\lambda={fmt(X4_LAMBDA, 1)}\\).

![x4_hist_exp](plots/x4_hist_exp.png)

### Гипотезы

```math
H_0:X_4\\sim Exp(\\lambda={fmt(X4_LAMBDA, 1)}),
```

```math
H_1:X_4 \\not\\sim Exp(\\lambda={fmt(X4_LAMBDA, 1)}).
```

Уровень значимости:

```math
\\alpha={ALPHA}.
```

### Выбор интервалов

Используем 6 интервалов равной теоретической вероятности. Границы выбраны по квантилям экспоненциального распределения:

```math
q_p=-\\frac{{\\ln(1-p)}}{{\\lambda}}.
```

Тогда для каждого интервала:

```math
p_k=\\frac16,\\qquad np_k=\\frac{{99}}6=16.5.
```

Ожидаемые частоты больше 5, поэтому объединение интервалов не требуется.

Ошибка первого рода: отвергнуть согласие X<sub>4</sub> с экспоненциальным распределением Exp({fmt(X4_LAMBDA, 1)}), хотя это распределение действительно является верной моделью.

### Таблица частот

{markdown_table(pearson_rows, ["Интервал", "n_k", "p_k", "np_k", "(n_k-np_k)^2/np_k"])}

### Статистика критерия

```math
\\chi^2_{{набл}}=\\sum_{{k=1}}^6\\frac{{(n_k-np_k)^2}}{{np_k}}={fmt(pearson['stat'])}.
```

Так как параметры распределения заданы заранее, число степеней свободы:

```math
\\nu=m-1=5.
```

Критическое значение:

```math
\\chi^2_{{5;0.95}}={fmt(pearson['crit'])}.
```

p-value:

```math
p={fmt(pearson['p_value'])}.
```

### Вывод

```math
\\chi^2_{{набл}}={fmt(pearson['stat'])}<{fmt(pearson['crit'])},\\qquad p={fmt(pearson['p_value'])}>0.05.
```

На уровне значимости 0.05 нет оснований отвергнуть гипотезу H<sub>0</sub>. Выборка X<sub>4</sub> не противоречит экспоненциальной модели с параметром \\(\\lambda={fmt(X4_LAMBDA, 1)}\\).

---

## 4.6. Итоговый вывод

В работе были проверены четыре гипотезы.

1. Для X<sub>1</sub> и X<sub>2</sub> параметрический t-критерий Стьюдента отверг гипотезу о равенстве математических ожиданий:

```math
p={fmt(pooled['p_value'])}<0.05.
```

2. Для X<sub>3</sub> одновыборочный t-критерий отверг гипотезу о математическом ожидании \\(\\mu_3={fmt(X3_MU0, 1)}\\):

```math
p={fmt(x3_test['p_value'])}<0.05.
```

3. Для X<sub>1</sub> и X<sub>2</sub> критерий Манна-Уитни также отверг гипотезу об однородности распределений:

```math
p={fmt(mw['p_value'])}<0.05.
```

4. Для X<sub>4</sub> критерий согласия Пирсона не отверг экспоненциальную модель Exp(\\(\\lambda={fmt(X4_LAMBDA, 1)}\\)):

```math
p={fmt(pearson['p_value'])}>0.05.
```

Содержательно результаты означают, что X<sub>2</sub> статистически значимо отличается от X<sub>1</sub> и имеет более высокие значения; X<sub>3</sub> по данной выборке заметно отклоняется от заявленного среднего {fmt(X3_MU0, 1)}; распределение X<sub>4</sub> согласуется с заявленной экспоненциальной моделью.
"""


def main():
    df = pd.read_csv(DATA_FILE, sep=";")
    desc = {col: describe(df[col]) for col in df.columns}
    pooled = pooled_t_test(df["X1"], df["X2"])
    x3_test = one_sample_t_test(df["X3"], X3_MU0)
    mw = mann_whitney_u_test(df["X1"], df["X2"])
    pearson = pearson_exp_test(df["X4"], X4_LAMBDA, bins_count=6)

    make_plots(df, desc)

    report = build_report(df, desc, pooled, x3_test, mw, pearson)
    Path("rgr2_github_math.md").write_text(report, encoding="utf-8")
    pycharm_report = report.replace("```math", "$$").replace("```", "$$")
    Path("rgr2.md").write_text(pycharm_report, encoding="utf-8")

    print("n =", len(df))
    print("pooled t:", {k: pooled[k] for k in ["stat", "df", "crit", "p_value"]})
    print("x3 t:", {k: x3_test[k] for k in ["stat", "df", "crit", "p_value"]})
    print("mann-whitney:", {k: mw[k] for k in ["u", "z", "p_value"]})
    print("pearson:", {k: pearson[k] for k in ["stat", "df", "crit", "p_value"]})
    print("written: rgr2_github_math.md")
    print("written: rgr2.md")


if __name__ == "__main__":
    main()
