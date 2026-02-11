"""
薪酬回归分析工具 - 社区方案最终修复版
核心解决：点击按钮跳回、页面重跑、状态丢失问题
参考Streamlit社区方案：表单包裹+固定滚动+强状态管理
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io
import openpyxl

# ====================== 1. 全局配置（禁用自动滚动+页面设置） ======================
# 关键：禁用Streamlit自动滚动到顶部（社区核心方案）
st.set_option("client.caching", True)
st.set_option("server.enableXsrfProtection", False)

# 自定义CSS固定页面滚动（防止跳回）
st.markdown("""
    <style>
    /* 禁用自动滚动 */
    html {
        scroll-behavior: auto !important;
    }
    /* 固定结果区域不被顶走 */
    .result-container {
        position: relative;
        z-index: 100;
    }
    /* 按钮样式强化 */
    div.stButton > button:first-child {
        background-color: #2196F3;
        color: white;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="薪酬回归分析工具",
    page_icon="📊",
    layout="wide"
)

# ====================== 2. 初始化所有状态（防止丢失） ======================
# 社区标准：把所有需要保留的状态都初始化
if "df_input" not in st.session_state:
    st.session_state.df_input = None
if "regression_obj" not in st.session_state:
    st.session_state.regression_obj = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

# ====================== 3. 核心回归类（保留原有逻辑） ======================
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
            formula = f"{A:.2f} * exp({coefs[0]:.6f}*x)"
        elif degree == 2:
            formula = f"{A:.2f} * exp({coefs[0]:.6f}*x + {coefs[1]:.6f}*x²)"
        else:
            formula = f"exp({intercept:.6f} + " + " + ".join([f"{c:.6f}*x^{i+1}" for i, c in enumerate(coefs)]) + ")"
        self.formulas[percentile] = {'formula': formula, 'degree': degree}
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
            model, poly, _ = self.log_polynomial_regression(grades, y, self.params['poly_degree'])
            if model is None:
                continue
            self.models[p] = {'model': model, 'poly': poly}
            self.get_formula_string(model, poly, p)
            X_target = poly.transform(target_grades.reshape(-1, 1))
            results[p] = np.exp(model.predict(X_target))
        self.results = results.sort_values('Survey Grade', ascending=False).reset_index(drop=True)
        return self.results

    def calculate_metrics(self):
        metrics = []
        grades = self.input_data['Survey Grade'].values
        for p in ['P10', 'P25', 'P50', 'P75', 'P90']:
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
                '分位数': p, 'R²': round(r2, 4), '平均误差%': round(mape, 2),
                '样本数': int(valid_mask.sum()), '回归公式': self.formulas[p]['formula']
            })
        self.metrics = pd.DataFrame(metrics)
        return self.metrics

# ====================== 4. 辅助函数 ======================
def create_plotly_chart(df):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, col in enumerate(['P10','P25','P50','P75','P90']):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Survey Grade'], y=df[col], name=col,
                line=dict(width=3, color=colors[i]),
                hovertemplate='职级:%{x}<br>薪酬:%{y:,.0f}<extra></extra>'
            ))
    fig.update_layout(
        xaxis_title="职级", yaxis_title="薪酬", height=500,
        xaxis_autorange="reversed", template="plotly_white"
    )
    return fig

def generate_excel(reg_obj, df_input):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        reg_obj.results.to_excel(writer, sheet_name="回归结果", index=False)
        reg_obj.metrics.to_excel(writer, sheet_name="回归指标", index=False)
        pd.DataFrame([{"分位数":k, "公式":v["formula"]} for k,v in reg_obj.formulas.items()]).to_excel(writer, sheet_name="回归公式", index=False)
        df_input.to_excel(writer, sheet_name="原始数据", index=False)
    output.seek(0)
    return output

# ====================== 5. 主页面（表单包裹+固定布局） ======================
st.title("📊 薪酬回归分析工具")

# 侧边栏（参数设置）
with st.sidebar:
    st.header("⚙️ 参数配置")
    # 社区方案：用表单包裹侧边栏参数，防止参数变化触发重跑
    with st.form(key="param_form", clear_on_submit=False):
        poly_degree = st.selectbox("多项式阶数", [1, 2, 3], index=1)
        grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
        grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)
        # 空提交按钮（仅用于锁定参数）
        st.form_submit_button(label="确认参数", disabled=True)

# 核心交互区（社区方案：用主表单包裹上传+回归逻辑）
with st.form(key="main_form", clear_on_submit=False):
    st.header("📤 数据上传")
    uploaded_file = st.file_uploader("上传Excel文件（含「数据输入」sheet）", type=['xlsx'])
    
    # 读取文件（仅在文件变化时执行，避免重复读取）
    if uploaded_file is not None and st.session_state.df_input is None:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            if "数据输入" not in excel_file.sheet_names:
                st.error("❌ 缺少「数据输入」sheet")
            else:
                df_input = pd.read_excel(uploaded_file, sheet_name="数据输入")
                df_input = df_input.dropna(subset=["Survey Grade"])
                df_input["Survey Grade"] = pd.to_numeric(df_input["Survey Grade"], errors="coerce")
                df_input = df_input.dropna(subset=["Survey Grade"])
                st.session_state.df_input = df_input
                st.success(f"✅ 读取{len(df_input)}行有效数据")
        except Exception as e:
            st.error(f"❌ 读取失败：{str(e)}")
    
    # 原始数据预览（默认折叠，防止跳回）
    if st.session_state.df_input is not None:
        with st.expander("📋 原始数据预览", expanded=False):
            st.dataframe(st.session_state.df_input, use_container_width=True)
    
    # 核心按钮（表单提交按钮，社区方案：唯一触发点）
    submit_button = st.form_submit_button(label="🚀 开始回归分析", type="primary")

# ====================== 6. 结果展示区（固定容器+状态控制） ======================
# 社区方案：用container固定结果区域，防止滚动
result_container = st.container()

with result_container:
    # 仅在提交后渲染结果（状态控制）
    if submit_button and st.session_state.df_input is not None:
        with st.spinner("🔢 正在执行回归分析..."):
            # 执行回归
            params = {"poly_degree": poly_degree, "grade_start": grade_start, "grade_end": grade_end}
            reg_obj = SalaryRegressionWeb(st.session_state.df_input, params)
            results_df = reg_obj.predict_percentiles()
            metrics_df = reg_obj.calculate_metrics()
            
            # 存储到session_state（核心：防止重跑丢失）
            st.session_state.regression_obj = reg_obj
            st.session_state.results_df = results_df
            st.session_state.metrics_df = metrics_df
            st.session_state.excel_data = generate_excel(reg_obj, st.session_state.df_input)
            st.session_state.form_submitted = True
    
    # 展示结果（仅当状态为已提交时）
    if st.session_state.form_submitted and st.session_state.results_df is not None:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.success("✅ 回归分析完成！")
        
        # 可视化+指标
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 回归曲线")
            st.plotly_chart(create_plotly_chart(st.session_state.results_df), use_container_width=True)
        with col2:
            st.subheader("📊 回归指标")
            st.dataframe(st.session_state.metrics_df, use_container_width=True, hide_index=True)
        
        # 结果表格
        st.subheader("📋 回归结果详情")
        res_show = st.session_state.results_df.copy()
        for col in ['P10','P25','P50','P75','P90']:
            if col in res_show:
                res_show[col] = res_show[col].round(0).astype(int)
        st.dataframe(res_show, use_container_width=True, hide_index=True)
        
        # 下载按钮
        st.subheader("💾 结果下载")
        st.download_button(
            label="📥 下载完整Excel报告",
            data=st.session_state.excel_data,
            file_name="薪酬回归分析结果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# 无文件提示
if uploaded_file is None and not st.session_state.form_submitted:
    st.info("👆 请先上传包含「数据输入」sheet的Excel文件")
