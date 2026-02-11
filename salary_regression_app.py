"""
薪酬分位值回归分析工具（终极稳定版）
核心：单文件+极简依赖+Streamlit无响应修复
部署：仅需该文件 + requirements.txt
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io

# ====================== 1. 全局常量（极简） ======================
REQUIRED_COL = "Survey Grade"
QUANTILE_COLS = ["P10", "P25", "P50", "P75", "P90"]

# ====================== 2. 基础配置 ======================
st.set_page_config(
    page_title="薪酬分位值回归分析",
    page_icon="📊",
    layout="wide"
)

# 仅保留2个核心状态（极简，避免冲突）
if "valid_df" not in st.session_state:
    st.session_state.valid_df = None
if "reg_result" not in st.session_state:
    st.session_state.reg_result = None

# 极简样式（仅保留核心）
st.markdown("""
<style>
.stButton>button {background: #2563eb; color: white; border-radius: 8px;}
.warning {color: #dc2626;}
.success {color: #059669;}
</style>
""", unsafe_allow_html=True)

# ====================== 3. 核心函数（稳定优先） ======================
def check_data(file):
    """极简数据校验，只做必要检查，避免复杂逻辑"""
    try:
        # 1. 检查Sheet
        if "数据输入" not in pd.ExcelFile(file).sheet_names:
            return False, "❌ 缺少「数据输入」工作表"
        
        # 2. 读取数据
        df = pd.read_excel(file, sheet_name="数据输入")
        
        # 3. 检查核心列
        if REQUIRED_COL not in df.columns:
            return False, f"❌ 缺少{REQUIRED_COL}列（职级）"
        if not [col for col in QUANTILE_COLS if col in df.columns]:
            return False, "❌ 缺少分位值列（P10/P25/P50/P75/P90）"
        
        # 4. 基础清洗（只做必要的）
        df[REQUIRED_COL] = pd.to_numeric(df[REQUIRED_COL], errors="coerce")
        df = df.dropna(subset=[REQUIRED_COL]).drop_duplicates(subset=[REQUIRED_COL])
        if len(df) < 3:
            return False, "❌ 有效职级数据不足3行"
        
        # 5. 分位值列清洗
        for col in QUANTILE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return True, df
    except Exception as e:
        return False, f"❌ 文件读取失败：{str(e)}"

def reg_analysis(df, poly_degree=2, grade_start=3, grade_end=21):
    """极简回归逻辑，只保留核心计算"""
    results = pd.DataFrame({REQUIRED_COL: np.arange(grade_start, grade_end+1)})
    metrics = []
    formulas = {}

    for col in [c for c in QUANTILE_COLS if c in df.columns]:
        valid = df.dropna(subset=[col])
        if len(valid) < 3:
            continue
        
        # 核心回归
        X = valid[REQUIRED_COL].values.reshape(-1,1)
        log_y = np.log(valid[col].values)
        X_poly = PolynomialFeatures(poly_degree).fit_transform(X)
        model = LinearRegression().fit(X_poly, log_y)
        
        # 预测
        y_pred = np.exp(model.predict(PolynomialFeatures(poly_degree).transform(results[REQUIRED_COL].values.reshape(-1,1))))
        results[col] = y_pred
        
        # 指标
        y_pred_train = np.exp(model.predict(X_poly))
        r2 = 1 - np.sum((valid[col].values - y_pred_train)**2) / np.sum((valid[col].values - valid[col].mean())**2)
        mape = np.mean(np.abs((valid[col].values - y_pred_train)/valid[col].values)) * 100
        
        # 公式
        A = np.exp(model.intercept_)
        coefs = model.coef_[1:]
        formula = f"{A:.2f} × e^({coefs[0]:.6f}x)" if poly_degree==1 else f"{A:.2f} × e^({coefs[0]:.6f}x + {coefs[1]:.6f}x²)"
        
        formulas[col] = formula
        metrics.append({"分位数":col, "R²":round(r2,4), "平均误差(%)":round(mape,2), "样本数":len(valid)})

    return {"results": results.sort_values(REQUIRED_COL, ascending=False).reset_index(drop=True),
            "metrics": pd.DataFrame(metrics), "formulas": formulas}

def make_excel(result, raw_df):
    """极简Excel生成，容错优先"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result["results"].to_excel(writer, "回归结果", index=False)
            result["metrics"].to_excel(writer, "回归指标", index=False)
            pd.DataFrame([{"分位数":k, "公式":v} for k,v in result["formulas"].items()]).to_excel(writer, "回归公式", index=False)
            raw_df.to_excel(writer, "原始数据", index=False)
        output.seek(0)
        return output
    except:
        return None

# ====================== 4. 页面主体（无响应修复） ======================
st.title("📊 薪酬分位值回归分析工具")

# 侧边栏参数（极简，无复杂校验）
with st.sidebar:
    st.subheader("⚙️ 参数")
    poly_degree = st.selectbox("多项式阶数", [1,2], index=1)
    grade_start = st.number_input("职级起始", value=3, min_value=1)
    grade_end = st.number_input("职级结束", value=21, min_value=1)

# 1. 文件上传（核心，稳定优先）
uploaded_file = st.file_uploader("📤 上传Excel文件", type=["xlsx"])
if uploaded_file:
    # 数据校验（极简提示）
    is_ok, res = check_data(uploaded_file)
    if not is_ok:
        st.markdown(f"<p class='warning'>{res}</p>", unsafe_allow_html=True)
    else:
        st.session_state.valid_df = res
        st.markdown("<p class='success'>✅ 数据校验通过！</p>", unsafe_allow_html=True)
        
        # 数据预览（极简）
        with st.expander("📋 数据预览", expanded=False):
            st.dataframe(res, use_container_width=True, hide_index=True)
        
        # 2. 分析按钮（无禁用，点击必响应）
        if st.button("🚀 生成回归分析", type="primary"):
            # 仅做基础参数检查，提示而非禁用
            if grade_start > grade_end:
                st.markdown("<p class='warning'>❌ 职级起始不能大于结束</p>", unsafe_allow_html=True)
            else:
                with st.spinner("分析中..."):
                    st.session_state.reg_result = reg_analysis(res, poly_degree, grade_start, grade_end)
                    st.success("✅ 分析完成！")

# 3. 结果展示（只依赖状态，不依赖按钮）
if st.session_state.reg_result is not None:
    res = st.session_state.reg_result
    
    # 无结果提示（无rerun，只文字提示）
    if len(res["formulas"]) == 0:
        st.markdown("<p class='warning'>⚠️ 无有效分位值数据</p>", unsafe_allow_html=True)
    else:
        # 回归曲线
        st.subheader("1. 回归曲线")
        fig = go.Figure()
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
        for i, (q, f) in enumerate(res["formulas"].items()):
            fig.add_trace(go.Scatter(x=res["results"][REQUIRED_COL], y=res["results"][q], name=q, line=dict(width=3, color=colors[i])))
        fig.update_layout(xaxis_title="职级", yaxis_title="薪酬", height=400, xaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        
        # 公式+指标
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. 回归公式")
            for q, f in res["formulas"].items():
                st.write(f"**{q}**：{f}")
        with col2:
            st.subheader("3. 回归指标")
            st.dataframe(res["metrics"], use_container_width=True, hide_index=True)
        
        # 结果表格（复制DataFrame，避免直接修改）
        st.subheader("4. 回归结果")
        show_df = res["results"].copy()
        for col in show_df.columns[1:]:
            show_df[col] = show_df[col].round(0).astype(int)
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        
        # 下载（极简容错）
        st.subheader("5. 下载报告")
        excel = make_excel(res, st.session_state.valid_df)
        if excel:
            st.download_button("📥 下载Excel", excel, "薪酬回归报告.xlsx", type="primary")

# 无文件提示
if not uploaded_file:
    st.info("👆 上传Excel文件（含「数据输入」sheet，列：Survey Grade + P10/P25/P50/P75/P90）")
    st.dataframe(pd.DataFrame({REQUIRED_COL:[3,4,5], "P50":[42486,52800,65400]}), use_container_width=True)
