"""
薪酬回归分析工具 - 最终稳定版
彻底解决：点击按钮无响应、结果消失、页面跳回问题
核心方案：用session_state持久化所有状态，结果渲染完全依赖状态而非按钮
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io
import openpyxl

# ====================== 1. 初始化所有session_state（关键！） ======================
if "step" not in st.session_state:
    st.session_state.step = "upload"  # upload -> preprocess -> regression
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "regression_results" not in st.session_state:
    st.session_state.regression_results = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "formulas" not in st.session_state:
    st.session_state.formulas = {}
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

# ====================== 2. 页面配置 ======================
st.set_page_config(
    page_title="薪酬回归分析工具",
    page_icon="📊",
    layout="wide"
)

# 自定义CSS防止自动滚动
st.markdown("""
    <style>
    html {
        scroll-behavior: auto !important;
    }
    .stButton > button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== 3. 核心回归类 ======================
class SalaryRegression:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.models = {}
        self.formulas = {}
        self.results = None
        self.metrics = None

    def log_poly_reg(self, X, y, degree):
        valid_mask = (~np.isnan(y)) & (y > 0)
        if valid_mask.sum() < 3:
            return None, None
        X_valid = X[valid_mask].reshape(-1, 1)
        y_valid = y[valid_mask]
        log_y = np.log(y_valid)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_valid)
        model = LinearRegression().fit(X_poly, log_y)
        return model, poly

    def fit(self):
        grades = self.df['Survey Grade'].values
        target_grades = np.arange(self.params['start'], self.params['end']+1)
        results = pd.DataFrame({'Survey Grade': target_grades})
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']

        for p in percentiles:
            if p not in self.df.columns:
                continue
            y = self.df[p].values
            model, poly = self.log_poly_reg(grades, y, self.params['degree'])
            if model is None:
                continue
            self.models[p] = {'model': model, 'poly': poly}
            # 生成公式
            intercept = model.intercept_
            coefs = model.coef_[1:]
            A = np.exp(intercept)
            if self.params['degree'] == 2:
                formula = f"{A:.2f} * exp({coefs[0]:.6f}*x + {coefs[1]:.6f}*x²)"
            else:
                formula = f"exp({intercept:.6f} + " + " + ".join([f"{c:.6f}*x^{i+1}" for i, c in enumerate(coefs)]) + ")"
            self.formulas[p] = formula
            # 预测
            X_target = poly.transform(target_grades.reshape(-1, 1))
            results[p] = np.exp(model.predict(X_target))

        self.results = results.sort_values('Survey Grade', ascending=False).reset_index(drop=True)
        return self.results

    def calculate_metrics(self):
        metrics = []
        grades = self.df['Survey Grade'].values
        for p in self.models:
            y_original = self.df[p].values
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
                '样本数': int(valid_mask.sum()), '回归公式': self.formulas[p]
            })
        self.metrics = pd.DataFrame(metrics)
        return self.metrics

# ====================== 4. 辅助函数 ======================
def create_chart(df):
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

def generate_excel(results, metrics, formulas, df_raw):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        results.to_excel(writer, sheet_name="回归结果", index=False)
        metrics.to_excel(writer, sheet_name="回归指标", index=False)
        pd.DataFrame([{"分位数":k, "公式":v} for k,v in formulas.items()]).to_excel(writer, sheet_name="回归公式", index=False)
        df_raw.to_excel(writer, sheet_name="原始数据", index=False)
    output.seek(0)
    return output

# ====================== 5. 主应用逻辑 ======================
st.title("📊 薪酬回归分析工具")

# 侧边栏：参数设置（用表单包裹，防止重跑）
with st.sidebar:
    st.header("⚙️ 参数设置")
    with st.form("param_form"):
        poly_degree = st.selectbox("多项式阶数", [1, 2, 3], index=1)
        grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
        grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)
        st.form_submit_button("确认参数", disabled=True)

# 主区域：分步骤渲染
# 步骤1：上传数据
if st.session_state.step == "upload":
    st.header("📤 步骤1：上传数据")
    uploaded_file = st.file_uploader("上传Excel文件（含「数据输入」sheet）", type=['xlsx'])
    
    if uploaded_file is not None:
        with st.spinner("读取数据中..."):
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name="数据输入")
                df_raw = df_raw.dropna(subset=["Survey Grade"])
                df_raw["Survey Grade"] = pd.to_numeric(df_raw["Survey Grade"], errors="coerce")
                df_raw = df_raw.dropna(subset=["Survey Grade"])
                st.session_state.df_raw = df_raw
                st.success(f"✅ 成功读取 {len(df_raw)} 行有效数据")
                
                # 显示原始数据（默认折叠）
                with st.expander("📋 原始数据预览", expanded=False):
                    st.dataframe(df_raw, use_container_width=True)
                
                # 下一步按钮：更新状态，进入预处理步骤
                if st.button("🔧 下一步：数据预处理", type="primary"):
                    st.session_state.step = "preprocess"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ 读取失败：{str(e)}")

# 步骤2：数据预处理
elif st.session_state.step == "preprocess":
    st.header("🔧 步骤2：数据预处理")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ 请先上传数据")
        if st.button("返回上传", type="secondary"):
            st.session_state.step = "upload"
            st.rerun()
        st.stop()
    
    # 预处理选项
    with st.form("preprocess_form"):
        st.subheader("预处理选项")
        missing_strategy = st.selectbox("缺失值处理", ["自动剔除", "均值填充", "中位数填充"])
        outlier_strategy = st.selectbox("异常值处理", ["保留", "自动剔除（3σ）", "替换为均值"])
        preprocess_submit = st.form_submit_button("✅ 执行预处理", type="primary")
    
    if preprocess_submit:
        with st.spinner("预处理中..."):
            df_processed = st.session_state.df_raw.copy()
            # 缺失值处理
            if missing_strategy == "自动剔除":
                df_processed = df_processed.dropna(subset=['P50'])
            elif missing_strategy == "均值填充":
                df_processed['P50'] = df_processed['P50'].fillna(df_processed['P50'].mean())
            else:
                df_processed['P50'] = df_processed['P50'].fillna(df_processed['P50'].median())
            # 异常值处理
            if outlier_strategy != "保留":
                salary_mean = df_processed['P50'].mean()
                salary_std = df_processed['P50'].std()
                lower = salary_mean - 3*salary_std
                upper = salary_mean + 3*salary_std
                if outlier_strategy == "自动剔除（3σ）":
                    df_processed = df_processed[(df_processed['P50'] >= lower) & (df_processed['P50'] <= upper)]
                else:
                    df_processed.loc[(df_processed['P50'] < lower) | (df_processed['P50'] > upper), 'P50'] = salary_mean
            
            st.session_state.df_processed = df_processed
            st.success(f"✅ 预处理完成：{len(df_processed)} 行数据")
            
            # 显示预处理后数据
            with st.expander("📋 预处理后数据预览", expanded=True):
                st.dataframe(df_processed[['Survey Grade', 'P10', 'P25', 'P50', 'P75', 'P90']], use_container_width=True)
            
            # 下一步按钮：更新状态，进入回归步骤
            if st.button("🚀 下一步：生成回归结果", type="primary"):
                st.session_state.step = "regression"
                st.rerun()
    
    # 返回按钮
    if st.button("返回上传", type="secondary"):
        st.session_state.step = "upload"
        st.rerun()

# 步骤3：回归分析
elif st.session_state.step == "regression":
    st.header("🚀 步骤3：回归分析")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ 请先完成数据预处理")
        if st.button("返回预处理", type="secondary"):
            st.session_state.step = "preprocess"
            st.rerun()
        st.stop()
    
    # 回归按钮：点击后更新状态，执行回归
    if st.button("✅ 生成回归结果", type="primary"):
        with st.spinner("回归分析中..."):
            params = {
                'degree': poly_degree,
                'start': grade_start,
                'end': grade_end
            }
            reg = SalaryRegression(st.session_state.df_processed, params)
            results = reg.fit()
            metrics = reg.calculate_metrics()
            excel_data = generate_excel(results, metrics, reg.formulas, st.session_state.df_raw)
            
            # 保存所有结果到session_state
            st.session_state.regression_results = results
            st.session_state.metrics = metrics
            st.session_state.formulas = reg.formulas
            st.session_state.excel_data = excel_data
    
    # 显示回归结果（只要状态存在，就一直显示，不会消失）
    if st.session_state.regression_results is not None:
        st.success("✅ 回归分析完成！")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 回归曲线")
            st.plotly_chart(create_chart(st.session_state.regression_results), use_container_width=True)
        
        with col2:
            st.subheader("📊 回归指标")
            st.dataframe(st.session_state.metrics, use_container_width=True, hide_index=True)
            st.subheader("🔢 回归公式")
            for p, f in st.session_state.formulas.items():
                st.code(f"{p}: y = {f}", language="python")
        
        st.subheader("📋 回归结果详情")
        res_show = st.session_state.regression_results.copy()
        for col in ['P10','P25','P50','P75','P90']:
            if col in res_show.columns:
                res_show[col] = res_show[col].round(0).astype(int)
        st.dataframe(res_show, use_container_width=True, hide_index=True)
        
        st.subheader("💾 下载结果")
        st.download_button(
            label="📥 下载完整Excel报告",
            data=st.session_state.excel_data,
            file_name="薪酬回归分析结果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # 返回按钮
    if st.button("返回预处理", type="secondary"):
        st.session_state.step = "preprocess"
        st.rerun()
