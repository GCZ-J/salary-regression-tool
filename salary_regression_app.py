"""
薪酬分位值回归分析工具
核心功能：上传职级分位值数据 → 自动校验 → 回归分析 → 可视化 + 报告下载
部署说明：需配合 requirements.txt 使用
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io
from openpyxl import Workbook

# ====================== 1. 基础配置 ======================
st.set_page_config(
    page_title="薪酬分位值回归分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 极简状态管理（仅保留核心状态）
if "valid_data" not in st.session_state:
    st.session_state.valid_data = None  # 校验后的有效数据
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None  # 回归分析结果

# 自定义样式（简洁美观）
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 8px; padding: 0.5rem 2rem;}
    .result-card {background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
    .warning-text {color: #dc2626; font-weight: 500;}
    .success-text {color: #059669; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# ====================== 2. 核心函数 ======================
def validate_and_preprocess(file):
    """
    数据校验+预处理：返回 (校验结果, 提示信息, 有效数据)
    """
    # 1. 校验文件结构
    try:
        excel_file = pd.ExcelFile(file)
        if "数据输入" not in excel_file.sheet_names:
            return False, ["❌ Excel文件缺少「数据输入」工作表"], None
        
        # 读取数据
        df = pd.read_excel(file, sheet_name="数据输入")
        tips = []
        
        # 2. 校验核心列
        required_col = "Survey Grade"
        quantile_cols = ["P10", "P25", "P50", "P75", "P90"]
        available_quantile_cols = [col for col in quantile_cols if col in df.columns]
        
        if required_col not in df.columns:
            tips.append(f"❌ 缺少核心列：{required_col}（职级）")
        if not available_quantile_cols:
            tips.append(f"❌ 缺少分位值列（至少包含P10/P25/P50/P75/P90中的一个）")
        
        # 3. 数据类型校验+清洗
        if required_col in df.columns:
            # 职级列清洗：转数值、去空、去重
            df[required_col] = pd.to_numeric(df[required_col], errors="coerce")
            df = df.dropna(subset=[required_col])
            df = df.drop_duplicates(subset=[required_col])
            
            if len(df) == 0:
                tips.append("❌ 职级列无有效数据（空值/非数字）")
        
        # 4. 分位值列清洗
        for col in available_quantile_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid_count = df[col].notna().sum()
            
            if valid_count == 0:
                tips.append(f"⚠️ {col}列无有效数值（空值/非数字），已跳过该列")
            elif valid_count < 3:
                tips.append(f"⚠️ {col}列有效样本仅{valid_count}个（至少需3个），已跳过该列")
        
        # 5. 最终数据筛选
        valid_df = df.copy()
        # 只保留有有效分位值的行
        if available_quantile_cols:
            valid_df = valid_df.dropna(subset=available_quantile_cols, how="all")
        
        # 校验结果判断
        is_valid = len([t for t in tips if t.startswith("❌")]) == 0
        if is_valid and len(valid_df) > 0:
            tips.append(f"✅ 数据校验通过！有效数据行数：{len(valid_df)}")
        elif not is_valid:
            valid_df = None
        
        return is_valid, tips, valid_df
    
    except Exception as e:
        return False, [f"❌ 文件读取失败：{str(e)}"], None

def run_salary_regression(df, poly_degree=2, grade_start=3, grade_end=21):
    """
    执行对数多项式回归
    返回：包含results/metrics/formulas的字典
    """
    # 准备基础数据
    required_col = "Survey Grade"
    quantile_cols = [col for col in ["P10", "P25", "P50", "P75", "P90"] if col in df.columns]
    target_grades = np.arange(grade_start, grade_end + 1)
    
    # 初始化结果容器
    results_df = pd.DataFrame({required_col: target_grades})
    metrics_list = []
    formulas_dict = {}
    
    # 逐个分位值回归
    for col in quantile_cols:
        # 筛选该列有效数据
        valid_data = df.dropna(subset=[col])
        if len(valid_data) < 3:
            continue
        
        X = valid_data[required_col].values.reshape(-1, 1)
        y = valid_data[col].values
        
        # 对数变换 + 多项式回归
        log_y = np.log(y)
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, log_y)
        
        # 预测
        X_target_poly = poly.transform(target_grades.reshape(-1, 1))
        y_pred_log = model.predict(X_target_poly)
        y_pred = np.exp(y_pred_log)
        results_df[col] = y_pred
        
        # 计算拟合指标
        y_pred_train_log = model.predict(X_poly)
        y_pred_train = np.exp(y_pred_train_log)
        r2 = 1 - np.sum((y - y_pred_train) ** 2) / np.sum((y - y.mean()) ** 2)
        mape = np.mean(np.abs((y - y_pred_train) / y)) * 100
        
        # 生成回归公式
        intercept = model.intercept_
        coefs = model.coef_[1:]  # 排除x^0的系数
        A = np.exp(intercept)
        
        if poly_degree == 1:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x)"
        elif poly_degree == 2:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x + {coefs[1]:.6f}x²)"
        else:
            formula = f"e^({intercept:.6f} + " + " + ".join([f"{c:.6f}x^{i+1}" for i, c in enumerate(coefs)]) + ")"
        
        # 保存结果
        formulas_dict[col] = formula
        metrics_list.append({
            "分位数": col,
            "R²": round(r2, 4),
            "平均误差(%)": round(mape, 2),
            "有效样本数": len(valid_data)
        })
    
    # 职级降序排列
    results_df = results_df.sort_values(required_col, ascending=False).reset_index(drop=True)
    
    return {
        "results": results_df,
        "metrics": pd.DataFrame(metrics_list),
        "formulas": formulas_dict
    }

def generate_excel_report(analysis_result, raw_data):
    """生成Excel分析报告"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 1. 回归结果
        analysis_result["results"].to_excel(writer, sheet_name="回归结果", index=False)
        # 2. 回归指标
        analysis_result["metrics"].to_excel(writer, sheet_name="回归指标", index=False)
        # 3. 回归公式
        formula_df = pd.DataFrame([
            {"分位数": k, "回归公式": v} for k, v in analysis_result["formulas"].items()
        ])
        formula_df.to_excel(writer, sheet_name="回归公式", index=False)
        # 4. 原始数据
        raw_data.to_excel(writer, sheet_name="原始数据", index=False)
    
    output.seek(0)
    return output

# ====================== 3. 页面主体 ======================
st.title("📊 薪酬分位值回归分析工具")
st.divider()

# 侧边栏参数配置
with st.sidebar:
    st.subheader("⚙️ 分析参数")
    poly_degree = st.selectbox("多项式阶数（推荐2阶）", [1, 2], index=1)
    grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
    grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)
    st.info("📌 数据要求：Excel文件需包含「数据输入」工作表，列包含Survey Grade（职级）+ P10/P25/P50/P75/P90（至少一个）")

# 1. 文件上传区域
uploaded_file = st.file_uploader("📤 上传Excel数据文件", type=["xlsx"], help="请上传包含职级分位值数据的Excel文件")

if uploaded_file:
    # 数据校验
    is_valid, tips, valid_df = validate_and_preprocess(uploaded_file)
    
    # 显示校验结果
    st.subheader("🔍 数据校验结果")
    for tip in tips:
        if tip.startswith("❌"):
            st.markdown(f"<p class='warning-text'>{tip}</p>", unsafe_allow_html=True)
        elif tip.startswith("✅"):
            st.markdown(f"<p class='success-text'>{tip}</p>", unsafe_allow_html=True)
        else:
            st.warning(tip)
    
    # 保存有效数据到状态
    if is_valid and valid_df is not None:
        st.session_state.valid_data = valid_df
        
        # 数据预览
        with st.expander("📋 有效数据预览", expanded=False):
            st.dataframe(valid_df, use_container_width=True, hide_index=True)
        
        # 参数合法性校验
        if grade_start > grade_end:
            st.markdown("<p class='warning-text'>❌ 错误：目标职级起始值不能大于结束值</p>", unsafe_allow_html=True)
        else:
            # 2. 分析按钮
            if st.button("🚀 一键生成回归分析", type="primary"):
                with st.spinner("正在执行回归分析，请稍候..."):
                    # 执行回归
                    analysis_result = run_salary_regression(
                        valid_df,
                        poly_degree=poly_degree,
                        grade_start=grade_start,
                        grade_end=grade_end
                    )
                    st.session_state.analysis_result = analysis_result
                    st.success("✅ 回归分析完成！")

# 3. 结果展示区域
if st.session_state.analysis_result is not None:
    st.divider()
    st.subheader("📈 回归分析结果")
    result = st.session_state.analysis_result
    
    # 无有效回归结果的处理
    if len(result["formulas"]) == 0:
        st.markdown("<p class='warning-text'>⚠️ 无有效分位值数据完成回归分析，请检查数据后重试</p>", unsafe_allow_html=True)
    else:
        # 3.1 回归曲线可视化
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("1. 回归曲线")
            fig = go.Figure()
            colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
            for idx, (quantile, formula) in enumerate(result["formulas"].items()):
                fig.add_trace(go.Scatter(
                    x=result["results"]["Survey Grade"],
                    y=result["results"][quantile],
                    name=quantile,
                    line=dict(width=3, color=colors[idx % len(colors)]),
                    hovertemplate="职级：%{x}<br>薪酬：%{y:,.0f}<extra></extra>"
                ))
            fig.update_layout(
                xaxis_title="职级",
                yaxis_title="薪酬",
                height=400,
                xaxis_autorange="reversed",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3.2 回归公式
        with col2:
            st.subheader("2. 回归公式")
            for quantile, formula in result["formulas"].items():
                st.markdown(f"""
                <div class='result-card'>
                    <strong>{quantile}</strong><br>
                    {formula}
                </div>
                """, unsafe_allow_html=True)
        
        # 3.3 回归指标
        st.subheader("3. 回归拟合指标")
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)
        
        # 3.4 回归结果详情
        st.subheader("4. 回归结果详情")
        display_df = result["results"].copy()
        # 薪酬数值格式化（取整）
        for col in display_df.columns:
            if col != "Survey Grade":
                display_df[col] = display_df[col].round(0).astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 3.5 报告下载
        st.subheader("5. 分析报告下载")
        excel_file = generate_excel_report(result, st.session_state.valid_data)
        st.download_button(
            label="📥 下载完整Excel分析报告",
            data=excel_file,
            file_name="薪酬分位值回归分析报告.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

# 无文件上传时的提示
if not uploaded_file:
    st.info("👆 请上传符合格式要求的Excel文件开始分析")
    # 示例数据格式展示
    st.subheader("📝 数据格式示例")
    sample_data = pd.DataFrame({
        "Survey Grade": [3, 4, 5, 6, 7],
        "P50": [42486, 52800, 65400, 78000, 94307],
        "P75": [47105, 55705, 69319, 85000, 106200]
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# 页脚
st.divider()
st.markdown("<p style='text-align:center; color:#64748b;'>薪酬分位值回归分析工具 | 基于Python+Streamlit构建</p>", unsafe_allow_html=True)
