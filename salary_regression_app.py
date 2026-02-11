"""
薪酬分位值回归分析工具 - 带完整数据校验版
功能：上传 → 自动校验 → 一键回归 → 显示结果+下载报告
校验：格式、列名、数据类型、样本量、参数合法性
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io

# ====================== 页面配置 ======================
st.set_page_config(page_title="薪酬分位值回归", page_icon="📊", layout="wide")

# 状态
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "output" not in st.session_state:
    st.session_state.output = {"results": None, "metrics": None, "formulas": None, "excel": None}

# ====================== 【核心】数据校验函数 ======================
def validate_data(df):
    """
    完整数据校验，返回 (是否通过, 错误信息)
    """
    errors = []

    # 1. 必须有 Survey Grade
    if "Survey Grade" not in df.columns:
        errors.append("❌ 缺少必选列：Survey Grade（职级）")
    else:
        # 转数字，去掉空值
        df["Survey Grade"] = pd.to_numeric(df["Survey Grade"], errors="coerce")
        if df["Survey Grade"].notna().sum() == 0:
            errors.append("❌ Survey Grade 列全为空或不是数字")

    # 2. 必须至少有一个分位值列
    quantile_cols = [c for c in ["P10", "P25", "P50", "P75", "P90"] if c in df.columns]
    if not quantile_cols:
        errors.append("❌ 未找到任何分位值列：P10/P25/P50/P75/P90")
    else:
        # 3. 每个分位值列检查
        for col in quantile_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid_cnt = df[col].notna().sum()
            if valid_cnt == 0:
                errors.append(f"❌ {col} 列全为空或不是数字")
            elif valid_cnt < 3:
                errors.append(f"⚠️ {col} 有效样本只有 {valid_cnt} 个，至少需要 3 个")

    # 4. 去重检查（职级不能重复）
    if "Survey Grade" in df.columns:
        dup = df["Survey Grade"].duplicated().sum()
        if dup > 0:
            errors.append(f"⚠️ 发现 {dup} 个重复职级，已自动去重")

    return len(errors) == 0, errors, df

# ====================== 预处理 ======================
def preprocess_data(df):
    df = df.dropna(subset=["Survey Grade"])
    df = df.drop_duplicates(subset=["Survey Grade"])
    for col in ["P10", "P25", "P50", "P75", "P90"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ====================== 回归逻辑 ======================
def run_regression(df, poly_degree=2, grade_start=3, grade_end=21):
    quantile_cols = [c for c in ["P10", "P25", "P50", "P75", "P90"] if c in df.columns]
    target_grades = np.arange(grade_start, grade_end + 1)
    results = pd.DataFrame({"Survey Grade": target_grades})
    metrics, formulas = [], {}

    for col in quantile_cols:
        valid_df = df.dropna(subset=[col])
        if len(valid_df) < 3:
            continue

        X = valid_df["Survey Grade"].values.reshape(-1, 1)
        y = valid_df[col].values
        log_y = np.log(y)
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, log_y)

        y_pred = np.exp(model.predict(poly.transform(target_grades.reshape(-1, 1))))
        results[col] = y_pred

        y_pred_train = np.exp(model.predict(X_poly))
        r2 = 1 - np.sum((y - y_pred_train) ** 2) / np.sum((y - y.mean()) ** 2)
        mape = np.mean(np.abs((y - y_pred_train) / y)) * 100

        intercept = model.intercept_
        coefs = model.coef_[1:]
        A = np.exp(intercept)
        if poly_degree == 2:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x + {coefs[1]:.6f}x²)"
        else:
            formula = f"{A:.2f} × e^({coefs[0]:.6f}x)"

        formulas[col] = formula
        metrics.append({"分位数": col, "R²": round(r2, 4), "平均误差%": round(mape, 2), "样本数": len(valid_df)})

    results = results.sort_values("Survey Grade", ascending=False)
    return {"results": results, "metrics": pd.DataFrame(metrics), "formulas": formulas}

# ====================== 生成Excel ======================
def generate_excel(results, metrics, formulas, raw_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="回归结果", index=False)
        metrics.to_excel(writer, sheet_name="回归指标", index=False)
        pd.DataFrame([{"分位数": k, "回归公式": v} for k, v in formulas.items()]).to_excel(writer, sheet_name="回归公式", index=False)
        raw_df.to_excel(writer, sheet_name="原始数据", index=False)
    output.seek(0)
    return output

# ====================== 主界面 ======================
st.title("📊 薪酬分位值回归分析（带数据校验）")

# 侧边参数
with st.sidebar:
    st.header("⚙️ 参数")
    poly_degree = st.selectbox("多项式阶数", [1, 2], index=1)
    grade_start = st.number_input("职级起始", value=3)
    grade_end = st.number_input("职级结束", value=21)

# 上传
uploaded_file = st.file_uploader("📤 上传 Excel（含「数据输入」sheet）", type=["xlsx"])

if uploaded_file:
    # 1. 读取并校验
    try:
        # 检查sheet是否存在
        excel_sheets = pd.ExcelFile(uploaded_file).sheet_names
        if "数据输入" not in excel_sheets:
            st.error("❌ Excel 中没有「数据输入」这个工作表")
            st.stop()

        # 读取
        df_raw = pd.read_excel(uploaded_file, sheet_name="数据输入")
        # 校验
        is_ok, err_list, df_checked = validate_data(df_raw)
        # 显示错误
        if err_list:
            for e in err_list:
                st.warning(e)
        # 不通过则停止
        if not is_ok:
            st.error("❌ 数据格式不满足要求，无法分析")
            st.stop()

        # 预处理
        st.session_state.df = preprocess_data(df_checked)
        st.success(f"✅ 数据校验通过！有效数据：{len(st.session_state.df)} 行")

        # 预览
        with st.expander("📋 查看数据", expanded=False):
            st.dataframe(st.session_state.df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 文件读取失败：{str(e)}")
        st.stop()

    # 参数校验
    if grade_start > grade_end:
        st.error("❌ 职级起始不能大于结束")
        st.stop()

    # 一键分析
    if st.button("🚀 一键生成回归分析", type="primary"):
        with st.spinner("分析中..."):
            res = run_regression(st.session_state.df, poly_degree, grade_start, grade_end)
            excel = generate_excel(res["results"], res["metrics"], res["formulas"], st.session_state.df)
            st.session_state.output = {**res, "excel": excel}
            st.session_state.analysis_done = True

    # 展示结果
    if st.session_state.analysis_done:
        st.success("✅ 分析完成")
        out = st.session_state.output

        # 图表
        st.subheader("📈 回归曲线")
        fig = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        for i, col in enumerate(out["formulas"]):
            fig.add_trace(go.Scatter(x=out["results"]["Survey Grade"], y=out["results"][col], name=col, line=dict(width=3, color=colors[i])))
        fig.update_layout(xaxis_title="职级", yaxis_title="薪酬", height=500, xaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        # 指标 + 公式
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 回归指标")
            st.dataframe(out["metrics"], use_container_width=True, hide_index=True)
        with col2:
            st.subheader("🔢 回归公式")
            for q, f in out["formulas"].items():
                st.code(f"{q}: {f}")

        # 结果表格
        st.subheader("📋 回归结果")
        show_df = out["results"].copy()
        for c in show_df.columns[1:]:
            show_df[c] = show_df[c].round(0).astype(int)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # 下载
        st.subheader("💾 下载报告")
        st.download_button("📥 下载Excel", out["excel"], "薪酬回归报告.xlsx")

else:
    st.info("👆 请上传Excel文件")
    st.code("""数据格式要求：
1. 必须有 sheet 名叫：数据输入
2. 必须有列：Survey Grade
3. 必须有列：P10/P25/P50/P75/P90 至少一个""")
