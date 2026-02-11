import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from io import BytesIO, StringIO
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="薪酬回归分析工具",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- 工具说明 --------------------------
st.title("💰 薪酬回归分析工具")
st.markdown("### 使用说明")
st.markdown("""
1. 下载标准模板并填写薪酬调研数据（必填：职级、薪酬值；可选：部门/城市/年份）
2. 上传填写好的CSV/Excel文件
3. 配置回归参数（分位值、多项式阶数）
4. 查看回归结果、可视化图表，并下载分析报告
""")
st.divider()

# -------------------------- 1. 标准模板下载 --------------------------
st.sidebar.header("📋 模板下载")
# 创建示例模板数据
template_data = pd.DataFrame({
    "职级": ["P1", "P2", "P3", "M1", "M2"],
    "薪酬值": [8000, 12000, 18000, 25000, 40000],
    "部门": ["技术", "技术", "技术", "管理", "管理"],
    "城市": ["北京", "北京", "北京", "北京", "北京"],
    "年份": [2025, 2025, 2025, 2025, 2025]
})

# 模板下载功能
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='薪酬数据模板')
    return output.getvalue()

template_excel = convert_df_to_excel(template_data)
st.sidebar.download_button(
    label="下载Excel模板",
    data=template_excel,
    file_name="薪酬调研数据模板.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -------------------------- 2. 数据上传与校验 --------------------------
st.sidebar.header("📤 数据上传")
uploaded_file = st.sidebar.file_uploader(
    "上传CSV/Excel文件",
    type=["csv", "xlsx"],
    help="请确保包含'职级'和'薪酬值'列，参考模板格式"
)

# 初始化数据变量
df_raw = None
df_processed = None
valid_data = False

if uploaded_file is not None:
    # 读取上传文件
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        # 核心字段校验
        required_cols = ['职级', '薪酬值']
        if all(col in df_raw.columns for col in required_cols):
            # 数据类型校验
            df_raw['薪酬值'] = pd.to_numeric(df_raw['薪酬值'], errors='coerce')
            
            # 基础数据展示
            st.subheader("📊 原始数据预览")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # 数据量限制检查（1000行）
            if len(df_raw) > 1000:
                st.warning("⚠️ 上传数据超过1000行，仅处理前1000行数据")
                df_raw = df_raw.head(1000)
            
            valid_data = True
        else:
            st.error(f"❌ 缺少核心字段！必须包含：{required_cols}")
            valid_data = False
            
    except Exception as e:
        st.error(f"❌ 数据读取失败：{str(e)}")
        valid_data = False

# -------------------------- 3. 数据预处理配置（侧边栏） --------------------------
if valid_data:
    st.sidebar.header("🔧 数据预处理")
    
    # 3.1 职级映射（文本转数值）
    st.sidebar.subheader("职级数值映射")
    unique_grades = sorted(df_raw['职级'].unique())
    grade_mapping = {}
    
    # 自动生成默认映射（按字母/数字排序，从1开始）
    for i, grade in enumerate(unique_grades):
        grade_mapping[grade] = st.sidebar.number_input(
            f"{grade} → 数值",
            value=i+1,
            min_value=1,
            step=1,
            key=f"grade_{grade}"
        )
    
    # 应用职级映射
    df_raw['职级数值'] = df_raw['职级'].map(grade_mapping)
    
    # 3.2 缺失值处理
    st.sidebar.subheader("缺失值处理")
    missing_strategy = st.sidebar.selectbox(
        "薪酬值缺失值处理方式",
        ["自动剔除", "均值填充", "中位数填充"],
        index=0
    )
    
    # 3.3 异常值处理
    st.sidebar.subheader("异常值处理")
    outlier_strategy = st.sidebar.selectbox(
        "薪酬值异常值处理方式",
        ["保留", "自动剔除（3σ）", "替换为均值"],
        index=0
    )
    
    # -------------------------- 4. 执行数据预处理 --------------------------
    st.sidebar.subheader("📝 执行预处理")
    if st.sidebar.button("开始预处理", type="primary"):
        df_processed = df_raw.copy()
        
        # 处理缺失值
        if missing_strategy == "自动剔除":
            df_processed = df_processed.dropna(subset=['薪酬值'])
        elif missing_strategy == "均值填充":
            mean_salary = df_processed['薪酬值'].mean()
            df_processed['薪酬值'] = df_processed['薪酬值'].fillna(mean_salary)
        elif missing_strategy == "中位数填充":
            median_salary = df_processed['薪酬值'].median()
            df_processed['薪酬值'] = df_processed['薪酬值'].fillna(median_salary)
        
        # 处理异常值（3σ原则）
        if outlier_strategy != "保留":
            salary_mean = df_processed['薪酬值'].mean()
            salary_std = df_processed['薪酬值'].std()
            lower_bound = salary_mean - 3 * salary_std
            upper_bound = salary_mean + 3 * salary_std
            
            if outlier_strategy == "自动剔除（3σ）":
                df_processed = df_processed[
                    (df_processed['薪酬值'] >= lower_bound) & 
                    (df_processed['薪酬值'] <= upper_bound)
                ]
            elif outlier_strategy == "替换为均值":
                df_processed.loc[
                    (df_processed['薪酬值'] < lower_bound) | 
                    (df_processed['薪酬值'] > upper_bound), 
                    '薪酬值'
                ] = salary_mean
        
        # 预处理结果展示
        st.subheader("🧹 预处理后数据预览")
        st.dataframe(df_processed.head(10), use_container_width=True)
        st.info(f"✅ 预处理完成：原始{len(df_raw)}行 → 处理后{len(df_processed)}行")
        
        # -------------------------- 5. 回归参数配置 --------------------------
        st.sidebar.header("📈 回归参数配置")
        
        # 5.1 分位值回归配置
        st.sidebar.subheader("分位值回归")
        default_quantiles = [0.25, 0.5, 0.75, 0.9]
        quantile_input = st.sidebar.text_input(
            "自定义分位值（逗号分隔，0-1之间）",
            value="0.25,0.5,0.75,0.9",
            help="例如：0.1,0.3,0.8 表示10/30/80分位"
        )
        # 解析分位值
        try:
            quantiles = [float(q.strip()) for q in quantile_input.split(',')]
            quantiles = [q for q in quantiles if 0 < q < 1]
            if not quantiles:
                quantiles = default_quantiles
                st.sidebar.warning("⚠️ 分位值输入无效，使用默认值：0.25,0.5,0.75,0.9")
        except:
            quantiles = default_quantiles
            st.sidebar.warning("⚠️ 分位值输入无效，使用默认值：0.25,0.5,0.75,0.9")
        
        # 5.2 多项式回归配置
        st.sidebar.subheader("多项式回归")
        poly_degree = st.sidebar.slider(
            "多项式阶数",
            min_value=1,
            max_value=5,
            value=2,
            help="建议使用2阶（避免过拟合），超过3阶需谨慎"
        )
        
        # -------------------------- 6. 执行回归分析 --------------------------
        st.sidebar.subheader("🚀 执行回归分析")
        if st.sidebar.button("生成回归结果", type="primary"):
            # 6.1 按职级分组计算分位值
            grade_quantiles = df_processed.groupby('职级数值')['薪酬值'].quantile(quantiles).unstack()
            grade_quantiles.columns = [f"{int(q*100)}分位值" for q in quantiles]
            grade_quantiles = grade_quantiles.reset_index()
            
            # 6.2 分位值回归计算（线性回归）
            quantile_regression_results = {}
            for col in grade_quantiles.columns:
                if "分位值" in col:
                    # 过滤空值
                    temp_df = grade_quantiles.dropna(subset=[col])
                    if len(temp_df) < 2:
                        continue
                    
                    X = temp_df['职级数值'].values.reshape(-1, 1)
                    y = temp_df[col].values
                    
                    # 线性回归
                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_pred = lr.predict(X)
                    
                    quantile_regression_results[col] = {
                        '系数': lr.coef_[0],
                        '截距': lr.intercept_,
                        'R²': r2_score(y, y_pred),
                        'MSE': mean_squared_error(y, y_pred),
                        '预测值': y_pred
                    }
            
            # 6.3 多项式回归计算
            X_poly = df_processed['职级数值'].values.reshape(-1, 1)
            y_poly = df_processed['薪酬值'].values
            
            # 生成多项式特征
            poly_features = PolynomialFeatures(degree=poly_degree)
            X_poly_transformed = poly_features.fit_transform(X_poly)
            
            # 拟合模型
            poly_model = LinearRegression()
            poly_model.fit(X_poly_transformed, y_poly)
            y_poly_pred = poly_model.predict(X_poly_transformed)
            
            poly_results = {
                '系数': poly_model.coef_,
                '截距': poly_model.intercept_,
                'R²': r2_score(y_poly, y_poly_pred),
                'MSE': mean_squared_error(y_poly, y_poly_pred),
                '预测值': y_poly_pred
            }
            
            # -------------------------- 7. 结果展示 --------------------------
            # 7.1 分位值结果
            st.subheader("📊 分位值计算结果")
            st.dataframe(grade_quantiles, use_container_width=True)
            
            # 7.2 分位值回归指标
            st.subheader("📈 分位值回归指标")
            quantile_metrics = pd.DataFrame({
                '分位值': list(quantile_regression_results.keys()),
                '回归系数': [v['系数'] for v in quantile_regression_results.values()],
                '截距': [v['截距'] for v in quantile_regression_results.values()],
                'R²': [v['R²'] for v in quantile_regression_results.values()],
                'MSE': [v['MSE'] for v in quantile_regression_results.values()]
            })
            st.dataframe(quantile_metrics, use_container_width=True)
            
            # 7.3 多项式回归指标
            st.subheader("🔄 多项式回归指标")
            poly_metrics = pd.DataFrame({
                '指标': ['阶数', '截距'] + [f'X^{i}' for i in range(1, poly_degree+1)] + ['R²', 'MSE'],
                '值': [poly_degree, poly_results['截距']] + list(poly_results['系数'][1:]) + [poly_results['R²'], poly_results['MSE']]
            })
            st.dataframe(poly_metrics, use_container_width=True)
            
            # 低R²提示
            if poly_results['R²'] < 0.3:
                st.warning("⚠️ 多项式回归R²＜0.3，数据无明显规律，结果仅供参考！")
            
            # 7.4 可视化展示
            st.subheader("📉 回归结果可视化")
            fig = go.Figure()
            
            # 添加原始数据散点
            fig.add_trace(go.Scatter(
                x=df_processed['职级数值'],
                y=df_processed['薪酬值'],
                mode='markers',
                name='原始数据',
                marker=dict(size=8, color='lightgray', opacity=0.7)
            ))
            
            # 添加分位值曲线
            for col in quantile_regression_results.keys():
                temp_df = grade_quantiles.dropna(subset=[col])
                X_plot = np.linspace(temp_df['职级数值'].min(), temp_df['职级数值'].max(), 100)
                y_plot = quantile_regression_results[col]['系数'] * X_plot + quantile_regression_results[col]['截距']
                
                fig.add_trace(go.Scatter(
                    x=X_plot,
                    y=y_plot,
                    mode='lines',
                    name=f'{col}回归曲线',
                    line=dict(dash='dash')
                ))
            
            # 添加多项式回归曲线
            X_poly_plot = np.linspace(X_poly.min(), X_poly.max(), 100).reshape(-1, 1)
            X_poly_plot_transformed = poly_features.transform(X_poly_plot)
            y_poly_plot = poly_model.predict(X_poly_plot_transformed)
            
            fig.add_trace(go.Scatter(
                x=X_poly_plot.flatten(),
                y=y_poly_plot,
                mode='lines',
                name=f'{poly_degree}阶多项式回归',
                line=dict(width=3, color='red')
            ))
            
            # 图表样式配置
            fig.update_layout(
                title='职级-薪酬回归分析图',
                xaxis_title='职级数值',
                yaxis_title='薪酬值',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # -------------------------- 8. 结果下载 --------------------------
            st.subheader("💾 结果下载")
            
            # 8.1 汇总所有结果到Excel
            def create_summary_excel():
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 原始数据
                    df_raw.to_excel(writer, index=False, sheet_name='原始数据')
                    # 预处理后数据
                    df_processed.to_excel(writer, index=False, sheet_name='预处理后数据')
                    # 分位值结果
                    grade_quantiles.to_excel(writer, index=False, sheet_name='分位值计算结果')
                    # 分位值回归指标
                    quantile_metrics.to_excel(writer, index=False, sheet_name='分位值回归指标')
                    # 多项式回归指标
                    poly_metrics.to_excel(writer, index=False, sheet_name='多项式回归指标')
                    
                    # 预测值数据
                    df_pred = df_processed.copy()
                    df_pred['多项式回归预测值'] = poly_model.predict(poly_features.transform(X_poly))
                    df_pred.to_excel(writer, index=False, sheet_name='薪酬预测值')
                return output.getvalue()
            
            # 8.2 下载按钮
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel下载
                summary_excel = create_summary_excel()
                st.download_button(
                    label="📥 下载完整分析报告（Excel）",
                    data=summary_excel,
                    file_name=f"薪酬回归分析报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # 图表下载（PNG）
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                st.download_button(
                    label="🖼️ 下载回归分析图表（PNG）",
                    data=img_bytes,
                    file_name=f"薪酬回归分析图_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            st.success("🎉 回归分析完成！所有结果已准备就绪，可下载保存。")

# -------------------------- 无数据时的提示 --------------------------
if not valid_data and uploaded_file is None:
    st.info("ℹ️ 请先从侧边栏下载模板，填写数据后上传开始分析")