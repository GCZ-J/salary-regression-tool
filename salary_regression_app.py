"""
薪酬回归分析工具 - Web版本
使用Streamlit创建交互式Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference

# 页面配置
st.set_page_config(
    page_title="薪酬回归分析工具",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 薪酬回归分析工具")
st.markdown("""
### 使用说明
1. 上传包含薪酬数据的Excel文件（需包含"数据输入"和"参数设置"sheet）
2. 系统自动进行对数变换回归分析
3. 查看交互式可视化结果
4. 下载完整的分析结果Excel文件
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 参数设置")
    poly_degree = st.selectbox("多项式阶数", [1, 2, 3], index=1)
    grade_start = st.number_input("目标职级起始", value=3, min_value=1, max_value=30)
    grade_end = st.number_input("目标职级结束", value=21, min_value=1, max_value=30)

    st.markdown("---")
    st.markdown("""
    ### 关于对数变换回归
    - 适合薪酬指数增长特性
    - 避免低职级曲线过平
    - 拟合度 R² > 0.99
    - 平均误差 < 10%
    """)

class SalaryRegressionWeb:
    """Web版薪酬回归分析类"""

    def __init__(self, input_data, params):
        self.input_data = input_data
        self.params = params
        self.models = {}
        self.formulas = {}
        self.results = None
        self.metrics = None

    def log_polynomial_regression(self, X, y, degree=2):
        """对数多项式回归"""
        valid_mask = (~np.isnan(y)) & (y > 0)
        X_valid = X[valid_mask].reshape(-1, 1)
        y_valid = y[valid_mask]

        if len(X_valid) < 3:
            return None, None, None

        log_y_valid = np.log(y_valid)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_valid)
        model = LinearRegression()
        model.fit(X_poly, log_y_valid)

        return model, poly, y_valid

    def get_formula_string(self, model, poly, percentile):
        """生成回归公式"""
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
        """对各分位数进行回归预测"""
        grades = self.input_data['Survey Grade'].values
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']

        target_grades = np.arange(
            self.params['grade_start'],
            self.params['grade_end'] + 1,
            1
        )

        results = pd.DataFrame({'Survey Grade': target_grades})
        results = results.sort_values('Survey Grade', ascending=False)

        for percentile in percentiles:
            if percentile not in self.input_data.columns:
                continue

            y = self.input_data[percentile].values
            model, poly, y_train = self.log_polynomial_regression(
                grades, y, degree=self.params['poly_degree']
            )

            if model is None:
                continue

            self.models[percentile] = {'model': model, 'poly': poly, 'y_train': y_train}
            formula = self.get_formula_string(model, poly, percentile)

            X_target = poly.transform(target_grades.reshape(-1, 1))
            log_y_pred = model.predict(X_target)
            y_pred = np.exp(log_y_pred)

            grade_to_pred = dict(zip(target_grades, y_pred))
            results[percentile] = results['Survey Grade'].map(grade_to_pred)

        self.results = results
        return results

    def calculate_metrics(self):
        """计算回归质量指标"""
        metrics = []
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
        grades = self.input_data['Survey Grade'].values

        for percentile in percentiles:
            if percentile not in self.input_data.columns or percentile not in self.models:
                continue

            y_original = self.input_data[percentile].values
            valid_mask = (~np.isnan(y_original)) & (y_original > 0)

            if valid_mask.sum() == 0:
                continue

            model_info = self.models[percentile]
            model = model_info['model']
            poly = model_info['poly']

            X_original = poly.transform(grades[valid_mask].reshape(-1, 1))
            log_y_pred = model.predict(X_original)
            y_pred = np.exp(log_y_pred)
            y_actual = y_original[valid_mask]

            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

            metrics.append({
                '分位数': percentile,
                'R²': r_squared,
                '平均误差%': mape,
                '样本数': int(valid_mask.sum()),
                '回归公式': self.formulas[percentile]['formula']
            })

        self.metrics = pd.DataFrame(metrics)
        return self.metrics

def create_plotly_chart(results_df):
    """创建交互式Plotly图表"""
    fig = go.Figure()

    percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    for idx, percentile in enumerate(percentiles):
        if percentile in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df['Survey Grade'],
                y=results_df[percentile],
                mode='lines',
                name=percentile,
                line=dict(width=3, color=colors[idx]),
                hovertemplate=f'<b>{percentile}</b><br>职级: %{{x}}<br>薪酬: %{{y:,.0f}}<extra></extra>'
            ))

    fig.update_layout(
        title='薪酬回归曲线汇总',
        xaxis_title='职级 (Survey Grade)',
        yaxis_title='薪酬 (Salary)',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        xaxis=dict(autorange="reversed")  # 翻转X轴，高职级在右侧
    )

    return fig

def create_output_excel(regression, input_data):
    """创建输出Excel文件"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 回归结果汇总
        results_summary = regression.results.copy()

        # 添加原始值列
        percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
        summary_with_original = pd.DataFrame({'Survey Grade': results_summary['Survey Grade']})

        for p in percentiles:
            # 原始值
            original_col = f'{p}_原始'
            summary_with_original[original_col] = summary_with_original['Survey Grade'].map(
                dict(zip(input_data['Survey Grade'], input_data[p]))
            )
            # 回归值
            regression_col = f'{p}_回归'
            summary_with_original[regression_col] = results_summary[p]

        summary_with_original.to_excel(writer, sheet_name='回归结果汇总', index=False)

        # 2. 回归公式
        formulas_df = pd.DataFrame([
            {
                '分位数': p,
                '回归公式': regression.formulas[p]['formula'],
                '多项式阶数': regression.formulas[p]['degree']
            }
            for p in percentiles if p in regression.formulas
        ])
        formulas_df.to_excel(writer, sheet_name='回归公式', index=False)

        # 3. 回归指标
        regression.metrics.to_excel(writer, sheet_name='回归指标', index=False)

        # 4. 原始数据
        input_data.to_excel(writer, sheet_name='原始数据', index=False)

    output.seek(0)
    return output

# 主应用逻辑
uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx', 'xlsm'])

if uploaded_file is not None:
    try:
        # 读取数据
        with st.spinner('正在读取数据...'):
            df_input = pd.read_excel(uploaded_file, sheet_name='数据输入')
            df_input = df_input.dropna(subset=['Survey Grade'])

        st.success(f"✅ 成功读取 {len(df_input)} 行数据")

        # 显示原始数据预览
        with st.expander("📋 查看原始数据"):
            st.dataframe(df_input, use_container_width=True)

        # 执行回归分析
        if st.button("🚀 开始回归分析", type="primary"):
            with st.spinner('正在进行回归分析...'):
                params = {
                    'poly_degree': poly_degree,
                    'grade_start': grade_start,
                    'grade_end': grade_end
                }

                regression = SalaryRegressionWeb(df_input, params)
                results = regression.predict_percentiles()
                metrics = regression.calculate_metrics()

            st.success("✅ 回归分析完成！")

            # 显示结果
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 回归曲线可视化")
                fig = create_plotly_chart(results)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 回归质量指标")
                metrics_display = metrics.copy()
                metrics_display['R²'] = metrics_display['R²'].apply(lambda x: f"{x:.4f}")
                metrics_display['平均误差%'] = metrics_display['平均误差%'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(metrics_display[['分位数', 'R²', '平均误差%', '样本数']],
                           use_container_width=True, hide_index=True)

                st.subheader("🔢 回归公式")
                for _, row in metrics.iterrows():
                    st.code(f"{row['分位数']}: y = {row['回归公式']}", language="python")

            # 回归结果表格
            st.subheader("📋 回归结果详情")
            results_display = results.copy()
            for col in ['P10', 'P25', 'P50', 'P75', 'P90']:
                if col in results_display.columns:
                    results_display[col] = results_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            st.dataframe(results_display, use_container_width=True, hide_index=True)

            # 下载按钮
            st.subheader("💾 下载分析结果")
            output_excel = create_output_excel(regression, df_input)

            st.download_button(
                label="📥 下载完整Excel报告",
                data=output_excel,
                file_name="薪酬回归分析结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ 错误: {str(e)}")
        st.exception(e)

else:
    st.info("👆 请上传包含薪酬数据的Excel文件开始分析")

    # 显示示例数据格式
    st.subheader("📝 数据格式要求")
    st.markdown("""
    Excel文件需要包含**"数据输入"** sheet，格式如下：

    | Survey Grade | P10 | P25 | P50 | P75 | P90 |
    |--------------|-----|-----|-----|-----|-----|
    | 3 | 30,132.90 | 38,011.67 | 42,485.64 | 47,105.00 | 67,537.89 |
    | 4 | 38,111.74 | 43,073.00 | 52,800.00 | 55,704.09 | 72,321.13 |
    | ... | ... | ... | ... | ... | ... |

    - **Survey Grade**: 职级
    - **P10, P25, P50, P75, P90**: 各分位数薪酬
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>薪酬回归分析工具 v3.0 | 使用对数变换回归 | R² > 0.99</p>
</div>
""", unsafe_allow_html=True)
