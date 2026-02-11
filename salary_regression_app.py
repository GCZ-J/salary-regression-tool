"""
薪酬回归分析工具 - 最终稳定版
修复：点击回归后页面跳回、无响应、可视化不显示、下载失效
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="薪酬回归分析工具",
    page_icon="📊",
    layout="wide"
)

# ====================== 状态缓存（关键！防止重跑跳页） ======================
if "regression_done" not in st.session_state:
    st.session_state.regression_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "regression" not in st.session_state:
    st.session_state.regression = None
if "df_input" not in st.session_state:
    st.session_state.df_input = None

# ====================== 标题 ======================
st.title("📊 薪酬回归分析工具")

# ====================== 侧边栏 ======================
with st.sidebar:
    st.header("⚙️ 参数设置")
    poly_degree = st.selectbox("多项式阶数", [1, 2, 3], index=1)
    grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
    grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)

# ====================== 核心类 ======================
class SalaryRegressionWeb:
    def __init__(self, input_data, params):
        self.input_data = input_data
        self.params = params
        self.models = {}
        self.formulas = {}
        self.results = None
        self.metrics = None

    def log_polynomial_regression(self, X, y, degree=2):
        valid_mask = (~np.isnan(y)) & (y > 0)
        if valid_mask.sum() < 3:
            return None, None, None

        X_valid = X[valid_mask].reshape(-1, 1)
        y_valid = y[valid_mask]
        log_y_valid = np.log(y_valid)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_valid)
        model = LinearRegression().fit(X_poly, log_y_valid)
        return model, poly, y_valid

    def get_formula_string(self, model, poly, percentile):
        degree = self.params['poly_degree']
        intercept = model.intercept_
        coefs = model.coef_[1:]
        A = np.exp(intercept)

        if degree == 1:
            b = coefs[0]
            formula = f"{A:.2f} * exp({b:.6f}*x)"
        elif degree == 2:
            b, c = coefs[0], coefs[1]
            formula = f"{A:.2f} * exp({b:.6f}*x + {c:.6f}*x²)"
        else:
            formula = f"exp({intercept:.6f} + " + " + ".join([f"{c:.6f}*x^{i+1}" for i, c in enumerate(coefs)]) + ")"

        self.formulas[percentile] = {
            'formula': formula,
            'intercept': intercept,
            'coefficients': coefs.tolist(),
            'degree': degree,
            'A': A
        }
        return formula

    def predict_percentiles(self):
        grades = self.input_data['Survey Grade'].values
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
        target_grades = np.arange(self.params['grade_start'], self.params['grade_end'] + 1, 1)
        results = pd.DataFrame({'Survey Grade': target_grades})

        for p in percentiles:
            if p not in self.input_data.columns:
                continue
            y = self.input_data[p].values
            model, poly, y_train = self.log_polynomial_regression(grades, y, self.params['poly_degree'])
            if model is None:
                continue

            self.models[p] = {'model': model, 'poly': poly}
            self.get_formula_string(model, poly, p)
            X_target = poly.transform(target_grades.reshape(-1, 1))
            results[p] = np.exp(model.predict(X_target))

        results = results.sort_values('Survey Grade', ascending=False).reset_index(drop=True)
        self.results = results
        return results

    def calculate_metrics(self):
        metrics = []
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
        grades = self.input_data['Survey Grade'].values

        for p in percentiles:
            if p not in self.models or p not in self.input_data.columns:
                continue

            y_original = self.input_data[p].values
            valid_mask = (~np.isnan(y_original)) & (y_original > 0)
            if valid_mask.sum() == 0:
                continue

            model = self.models[p]['model']
            poly = self.models[p]['poly']
            X_org = poly.transform(grades[valid_mask].reshape(-1, 1))
            y_pred = np.exp(model.predict(X_org))
            y_actual = y_original[valid_mask]

            r2 = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - y_actual.mean())**2)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

            metrics.append({
                '分位数': p,
                'R²': r2,
                '平均误差%': mape,
                '样本数': int(valid_mask.sum()),
                '回归公式': self.formulas[p]['formula']
            })
        self.metrics = pd.DataFrame(metrics)
        return self.metrics

# ====================== 绘图 ======================
def create_plotly_chart(df):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, col in enumerate(['P10','P25','P50','P75','P90']):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['Survey Grade'], y=df[col], name=col, line=dict(width=3, color=colors[i])))
    fig.update_layout(xaxis_title="职级", yaxis_title="薪酬", height=500, xaxis_autorange="reversed")
    return fig

# ====================== 导出Excel ======================
def create_output_excel(reg, df_input):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        reg.results.to_excel(writer, sheet_name="回归结果", index=False)
        reg.metrics.to_excel(writer, sheet_name="回归指标", index=False)
        pd.DataFrame([{"分位数":k, "公式":v["formula"]} for k,v in reg.formulas.items()]).to_excel(writer, sheet_name="回归公式", index=False)
        df_input.to_excel(writer, sheet_name="原始数据", index=False)
    output.seek(0)
    return output

# ====================== 上传文件 ======================
uploaded_file = st.file_uploader("上传 Excel（必须含：数据输入 sheet）", type=['xlsx'])

if uploaded_file is not None:
    # 只读取一次，存到 session_state，防止重跑
    if st.session_state.df_input is None:
        df_input = pd.read_excel(uploaded_file, sheet_name="数据输入")
        df_input = df_input.dropna(subset=["Survey Grade"])
        df_input["Survey Grade"] = pd.to_numeric(df_input["Survey Grade"], errors="coerce")
        df_input = df_input.dropna(subset=["Survey Grade"])
        st.session_state.df_input = df_input

    # 【关键】默认折叠，防止点击回归后页面跳回这里
    with st.expander("📋 原始数据预览", expanded=False):
        st.dataframe(st.session_state.df_input, use_container_width=True)

    # ====================== 回归按钮 ======================
    if st.button("🚀 开始回归分析", type="primary"):
        with st.spinner("计算中..."):
            params = {"poly_degree": poly_degree, "grade_start": grade_start, "grade_end": grade_end}
            reg = SalaryRegressionWeb(st.session_state.df_input, params)
            results = reg.predict_percentiles()
            metrics = reg.calculate_metrics()

            # 存到状态里，不丢失
            st.session_state.regression = reg
            st.session_state.results = results
            st.session_state.metrics = metrics
            st.session_state.regression_done = True

    # ====================== 展示结果（只在状态完成后渲染） ======================
    if st.session_state.regression_done:
        res = st.session_state.results
        met = st.session_state.metrics
        reg = st.session_state.regression

        st.success("✅ 回归完成")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 回归曲线")
            st.plotly_chart(create_plotly_chart(res), use_container_width=True)

        with col2:
            st.subheader("📊 拟合效果")
            st.dataframe(met, use_container_width=True, hide_index=True)

        st.subheader("📋 回归结果")
        res_show = res.copy()
        for c in ['P10','P25','P50','P75','P90']:
            if c in res_show:
                res_show[c] = res_show[c].round(0).astype(int)
        st.dataframe(res_show, use_container_width=True, hide_index=True)

        # 下载
        st.subheader("💾 下载报告")
        xl = create_output_excel(reg, st.session_state.df_input)
        st.download_button("📥 下载 Excel 结果", xl, "薪酬回归结果.xlsx")

else:
    st.info("👈 请上传 Excel 文件（含 数据输入 sheet）")
