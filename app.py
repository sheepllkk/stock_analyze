import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import GradientBoostingClassifier

# 1. 网页配置
st.set_page_config(page_title="A股智能预测终端", layout="wide")
st.title("📈 A股智能量化分析终端")

# 2. 侧边栏交互
st.sidebar.header("参数设置")
ticker = st.sidebar.text_input("请输入股票代码 (如 600519.SS):", "600519.SS")
submit = st.sidebar.button("开始分析")

if submit:
    # 3. 数据拉取
    df = yf.download(ticker, period='5y', progress=False)
    if df.empty:
        st.error("无法获取数据，请检查代码格式！")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['Close'] = df['Close'].squeeze()
        df['Volume'] = df['Volume'].squeeze()

        # 4. 指标计算
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['Upper_Band'] = bb.bollinger_hband()
        df['Lower_Band'] = bb.bollinger_lband()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD_Line'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['Momentum_5D'] = df['Close'].pct_change(periods=5)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 5. 模型处理
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        ml_df = df.dropna().copy()

        X = ml_df[['Close', 'Upper_Band', 'Lower_Band', 'RSI', 'MACD_Line', 'MACD_Signal', 'Momentum_5D', 'Volume_Change']]
        y = ml_df['Target']
        
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X, y) # 这里为简化演示，用全部数据训练
        
        prediction = model.predict(X.iloc[-1:])

        # 6. 网页渲染展示
        st.subheader(f"分析结果: {ticker}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最新收盘价", f"￥{df['Close'].iloc[-1]:.2f}")
        with col2:
            direction = "上涨" if prediction[0] == 1 else "下跌"
            st.metric("次日预测方向", direction)
        
        st.line_chart(df['Close'])
        # --- 新增：AI 分析逻辑 ---
        st.divider()
        st.subheader("🤖 AI 结合新闻研判")
        
        # 1. 增加一个输入 API Key 的区域 (在侧边栏加一个输入框)
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            st.error("未在后台检测到 API Key，请联系管理员。")
            st.stop() # 如果没 Key，直接停止运行
        
        if api_key:
            with st.spinner("AI 正在根据新闻与指标深度思考..."):
                from openai import OpenAI
                
                # 获取新闻
                stock = yf.Ticker(ticker)
                news = stock.news
                news_text = "\n".join([item.get('title', '') for item in news[:5]])
                
                # 调用 AI
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                prompt = f"""
                分析股票 {ticker}：
                1. 量化模型预测结果：{"看涨" if prediction[0] == 1 else "看跌"}
                2. 最新新闻：{news_text}
                3. 技术指标：RSI={df['RSI'].iloc[-1]:.2f}
                
                请给出简洁的投资建议，包括看涨/看跌的逻辑理由。
                """
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # 网页显示 AI 结果
                st.write(response.choices[0].message.content)
        else:
            st.warning("请在左侧边栏输入 API Key 以查看 AI 深度研判。")
        st.success("分析完成！")
