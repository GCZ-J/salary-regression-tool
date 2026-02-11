"""
薪酬分位值回归分析工具（极简部署最终版）
核心：单文件+极简依赖，支持上传→校验→回归→可视化→报告下载
部署：仅需该文件 + requirements.txt
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io

# ====================== 1. 全局常量（精简冗余定义） ======================
REQUIRED_COL = "Survey Grade"  # 核心列常量
QUANTILE_COLS = ["P10", "P25", "P50", "P75", "P90"]  # 分位值列常量
PREVIEW_ROWS = 100  # 数据预览最大行数
PAGE_CONFIG = {
    "page_title": "薪酬分位值回归分析",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# ====================== 2. 基础配置 ======================
st.set_page_config(**PAGE_CONFIG)

# 极简状态管理（仅2个核心状态）
if "valid_data" not in st.session_state:
    st.session_state.valid_data = None  # 校验后的有效数据
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None  # 回归分析结果

# 极简样式（保留核心样式，减少冗余）
st.markdown("""
<style>
.stButton>button {background: #2563eb; color: white; border-radius: 8px; padding: 0.5rem 2rem;}
.result-card {background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
.warning-text {color: #dc2626; font-weight: 500;}
.success-text {color: #059669; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# ====================== 3. 核心函数（强化容错+精简代码） ======================
def validate_and_preprocess(file):
    """数据校验+预处理：返回 (校验结果, 提示信息, 有效数据)"""
    try:
        # 1. 校验Sheet
        excel_file = pd.ExcelFile(file)
        if "数据输入" not in excel_file.sheet_names:
            return False, ["❌ Excel文件缺少「数据输入」工作表"], None
        
        # 2. 读取并清洗数据
        df = pd.read_excel(file, sheet_name="数据输入")
        tips = []
        
        # 3. 核心列校验
        if REQUIRED_COL not in df.columns:
            tips.append(f"❌ 缺少核心列：{REQUIRED_COL}（职级）")
        else:
            df[REQUIRED_COL] = pd.to_numeric(df[REQUIRED_COL], errors="coerce")
            df = df.dropna(subset=[REQUIRED_COL]).drop_duplicates(subset=[REQUIRED_COL])
            if len(df) == 0:
                tips.append("❌ 职级列无有效数据（空值/非数字/重复）")
        
        # 4. 分位值列校验
        available_quantile = [col for col in QUANTILE_COLS if col in df.columns]
        if not available_quantile:
            tips.append("❌ 缺少分位值列（至少包含P10/P25/P50/P75/P90中的一个）")
        else:
            for col in available_quantile:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                valid_cnt = df[col].notna().sum()
                if valid_cnt == 0:
                    tips.append(f"⚠️ {col}列无有效数值，已跳过")
                elif valid_cnt < 3:
                    tips.append(f"⚠️ {col}列有效样本仅{valid_cnt}个（需≥3个），已跳过")
        
        # 5. 最终校验结果
        is_valid = len([t for t in tips if t.startswith("❌")]) == 0
        valid_df = df if is_valid and len(df) > 0 else None
        if is_valid and valid_df is not None:
            tips.append(f"✅ 数据校验通过！有效行数：{len(valid_df)}")
        
        return is_valid, tips, valid_df
    
    except Exception as e:
        return False, [f"❌ 文件读取失败：{str(e)}"], None

def run_salary_reg(df, poly_degree=2, grade_start=3, grade_end=21):
    """执行对数多项式回归（精简命名，强化边界）"""
    # 初始化结果
    target_grades = np.arange(grade_start, grade_end + 1)
    results_df = pd.DataFrame({REQUIRED_COL: target_grades})
    metrics, formulas = [], {}

    # 逐个分位值回归
    for col in [c for c in QUANTILE_COLS if c in df.columns]:
        valid_data = df.dropna(subset=[col])
        if len(valid_data) < 3:
            continue
        
        # 核心回归逻辑
        X = valid_data[REQUIRED_COL].values.reshape(-1, 1)
        log_y = np.log(valid_data[col].values)
        X_poly = PolynomialFeatures(degree=poly_degree).fit_transform(X)
        model = LinearRegression().fit(X_poly, log_y)
        
        # 预测+指标计算
        y_pred = np.exp(model.predict(PolynomialFeatures(degree=poly_degree).transform(target_grades.reshape(-1, 1))))
        y_pred_train = np.exp(model.predict(X_poly))
        r2 = 1 - np.sum((valid_data[col].values - y_pred_train)**2) / np.sum((valid_data[col].values - valid_data[col].mean())**2)
        mape = np.mean(np.abs((valid_data[col].values - y_pred_train) / valid_data[col].values)) * 100
        
        # 公式生成
        A = np.exp(model.intercept_)
        coefs = model.coef_[1:]
        if poly_degree == 1:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x)"
        else:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x + {coefs[1]:.6f}x²)"
        
        # 保存结果
        results_df[col] = y_pred
        formulas[col] = formula
        metrics.append({
            "分位数": col, "R²": round(r2, 4), "平均误差(%)": round(mape, 2), "有效样本数": len(valid_data)
        })

    return {
        "results": results_df.sort_values(REQUIRED_COL, ascending=False).reset_index(drop=True),
        "metrics": pd.DataFrame(metrics),
        "formulas": formulas
    }

def generate_excel_report(analysis_result, raw_data):
    """生成Excel报告（强化容错）"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            analysis_result["results"].to_excel(writer, sheet_name="回归结果", index=False)
            analysis_result["metrics"].to_excel(writer, sheet_name="回归指标", index=False)
            pd.DataFrame([{"分位数": k, "回归公式": v} for k, v in analysis_result["formulas"].items()]).to_excel(writer, sheet_name="回归公式", index=False)
            raw_data.to_excel(writer, sheet_name="原始数据", index=False)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"❌ 报告生成失败：{str(e)}")
        return None

# ====================== 4. 页面主体（优化交互+边界处理） ======================
st.title("📊 薪酬分位值回归分析工具")
st.divider()

# 侧边栏参数
with st.sidebar:
    st.subheader("⚙️ 分析参数")
    poly_degree = st.selectbox("多项式阶数（推荐2阶）", [1, 2], index=1)
    grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
    grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)
    st.info("📌 数据要求：Excel含「数据输入」sheet，列包含Survey Grade + P10/P25/P50/P75/P90（至少一个）")

# 文件上传
uploaded_file = st.file_uploader("📤 上传Excel数据文件", type=["xlsx"])
if uploaded_file:
    # 数据校验
    is_valid, tips, valid_df = validate_and_preprocess(uploaded_file)
    
    # 展示校验结果
    st.subheader("🔍 数据校验结果")
    for tip in tips:
        if tip.startswith("❌"):
            st.markdown(f"<p class='warning-text'>{tip}</p>", unsafe_allow_html=True)
        elif tip.startswith("✅"):
            st.markdown(f"<p class='success-text'>{tip}</p>", unsafe_allow_html=True)
        else:
            st.warning(tip)
    
    # 有效数据处理
    if is_valid and valid_df is not None:
        st.session_state.valid_data = valid_df
        
        # 数据预览（限制行数，优化性能）
        with st.expander("📋 有效数据预览", expanded=False):
            st.dataframe(valid_df.head(PREVIEW_ROWS), use_container_width=True, hide_index=True)
        
        # 按钮状态：参数合法才启用
        btn_disabled = grade_start > grade_end
        btn_text = "🚀 一键生成回归分析" if not btn_disabled else "❌ 职级起始不能大于结束"
        
        # 分析按钮
        if st.button(btn_text, type="primary", disabled=btn_disabled):
            with st.spinner("分析中..."):
                st.session_state.analysis_result = run_salary_reg(valid_df, poly_degree, grade_start, grade_end)
                st.success("✅ 回归分析完成！")

# 结果展示（强化边界处理）
if st.session_state.analysis_result is not None:
    st.divider()
    st.subheader("📈 回归分析结果")
    result = st.session_state.analysis_result
    
    # 无有效回归结果的友好处理
    if len(result["formulas"]) == 0:
        st.markdown("<p class='warning-text'>⚠️ 无有效分位值数据完成回归（样本量均<3个）</p>", unsafe_allow_html=True)
        if st.button("🔙 返回重新上传"):
            st.session_state.valid_data = None
            st.session_state.analysis_result = None
            st.rerun()
    else:
        # 1. 回归曲线
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("1. 回归曲线")
            fig = go.Figure()
            colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
            for idx, (q, f) in enumerate(result["formulas"].items()):
                fig.add_trace(go.Scatter(x=result["results"][REQUIRED_COL], y=result["results"][q], name=q, line=dict(width=3, color=colors[idx]), hovertemplate="职级：%{x}<br>薪酬：%{y:,.0f}"))
            fig.update_layout(xaxis_title="职级", yaxis_title="薪酬", height=400, xaxis_autorange="reversed", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        # 2. 回归公式
        with col2:
            st.subheader("2. 回归公式")
            for q, f in result["formulas"].items():
                st.markdown(f"<div class='result-card'><strong>{q}</strong><br>{f}</div>", unsafe_allow_html=True)
        
        # 3. 指标+结果
        st.subheader("3. 回归拟合指标")
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)
        
        st.subheader("4. 回归结果详情")
        # 优化：无需复制DataFrame，直接格式化
        result["results"].loc[:, result["results"].columns != REQUIRED_COL] = result["results"].loc[:, result["results"].columns != REQUIRED_COL].round(0).astype(int)
        st.dataframe(result["results"], use_container_width=True, hide_index=True)
        
        # 4. 报告下载（容错）
        st.subheader("5. 分析报告下载")
        excel_file = generate_excel_report(result, st.session_state.valid_data)
        if excel_file is not None:
            st.download_button("📥 下载Excel报告", excel_file, "薪酬回归分析报告.xlsx", type="primary")

# 无文件上传提示
if not uploaded_file:
    st.info("👆 请上传符合格式要求的Excel文件开始分析")
    st.subheader("📝 数据格式示例")
    st.dataframe(pd.DataFrame({REQUIRED_COL: [3,4,5], "P50": [42486, 52800, 65400]}), use_container_width=True, hide_index=True)
