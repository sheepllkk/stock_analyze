import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import GradientBoostingClassifier
from openai import OpenAI

# 1. 网页配置
st.set_page_config(page_title="A股智能预测终端", layout="wide")
st.title("📈 A股智能量化分析终端")

# 2. 侧边栏交互
st.sidebar.header("参数设置")
ticker = st.sidebar.text_input("请输入股票代码 (如 600519.SS):", "600519.SS")
submit = st.sidebar.button("开始分析")

if submit:
    # 3. 数据拉取
    with st.spinner("正在获取数据并训练模型..."):
        df = yf.download(ticker, period='5y', progress=False)
        if df.empty:
            st.error("无法获取数据，请检查代码格式！")
            st.stop()
        
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
        model.fit(X, y)
        prediction = model.predict(X.iloc[-1:])

        # 6. 网页渲染展示
        st.subheader(f"分析结果: {ticker}")
        col1, col2 = st.columns(2)
        latest_close = df['Close'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        
        col1.metric("最新收盘价", f"￥{latest_close:.2f}")
        col2.metric("次日预测方向", "上涨" if prediction[0] == 1 else "下跌")
        
        st.line_chart(df['Close'])

        # 7. AI 深度研判 (自动从 Secrets 读取)
        st.divider()
        st.subheader("🤖 AI 结合新闻研判")
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            with st.spinner("AI 正在根据新闻与指标深度思考..."):
                stock = yf.Ticker(ticker)
                news = stock.news
                news_text = "\n".join([item.get('title', '') for item in news[:5]]) if news else "近期暂无重大新闻。"
                
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                prompt = f"""
                分析股票 {ticker}：
                1. 量化模型预测结果：{"看涨" if prediction[0] == 1 else "看跌"}
                2. 当前收盘价：{latest_close:.2f}，RSI：{latest_rsi:.2f}
                3. 最新新闻：{news_text}
                
                请给出简洁的投资建议，包括看涨/看跌的逻辑理由，结构化输出。
                """
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
        except KeyError:
            st.error("未配置 API Key，请在 Streamlit Cloud 后台设置 OPENAI_API_KEY")
        except Exception as e:
            st.error(f"AI 分析出错: {e}")

        st.success("分析完成！")
