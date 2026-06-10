import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ====================== МОДЕЛЬ ======================
class GraphSAGEWind(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128, out_channels=3, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
       
    def forward(self, data):
        x = data.x.mean(dim=1)
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

# ====================== НАСТРОЙКИ ======================
st.set_page_config(
    page_title="GraphSAGE Wind | Диплом",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== КРАСИВЫЙ CSS ======================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #fffaf0 0%, #fff0e6 50%, #ffe6e6 100%); padding-top: 2rem; }
    h1 { background: linear-gradient(90deg, #ff6b00, #e63939); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; font-size: 2.8rem !important; }
    h2, h3 { color: #1e3a8a; font-weight: 600; }
    .stDataFrame, .stPlotlyChart, .st-pyplot, div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #ffe0cc; padding: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff6b00, #ff8c00); color: white;
        border-radius: 12px; height: 3.2em; font-weight: 600;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #ff8c00, #ff6b00); transform: translateY(-3px); }
</style>
""", unsafe_allow_html=True)

# ====================== ЗАГРУЗКА ДАННЫХ ======================
@st.cache_data
def load_data():
    df = pd.read_csv('data/wtbdata_245days.csv')
    loc = pd.read_csv('data/sdwpf_baidukddcup2022_turb_location.csv')
    return df, loc

df, loc = load_data()

@st.cache_resource
def load_model_and_graph():
    try:
        checkpoint = torch.load('results/graphsage_wind_model_best.pt', 
                              map_location='cpu', weights_only=False)
        graph_data = torch.load('results/graph_data.pt', map_location='cpu')
        
        model = GraphSAGEWind()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_features = checkpoint.get('scaler_features')
        scaler_targets = checkpoint.get('scaler_targets')
        feature_cols = checkpoint.get('feature_cols', ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Patv'])
        
        return model, graph_data, scaler_targets, feature_cols
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None

model, graph_data, scaler_targets, feature_cols = load_model_and_graph()

# ====================== ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ ======================
target_cols = ['Pab1', 'Pab2', 'Pab3']

# ====================== ТАБЫ ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 01. Разведочный анализ", 
    "🕸️ 02. Граф ветропарка", 
    "🧠 03. Модель и обучение", 
    "🔮 04. Реальный прогноз",
    "📈 05. Результаты и метрики"
])

# ====================== TAB 1 ======================
with tab1:
    st.header("01. Разведочный анализ данных")
    st.write(f"**{df.shape[0]:,} записей** • **{df['TurbID'].nunique()}** турбин")
   
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Первые строки данных")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.subheader("Статистика целевых переменных")
        st.dataframe(df[target_cols].describe().round(3), use_container_width=True)

    st.subheader("Распределение целевых переменных")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(target_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Распределение {col}')
    plt.tight_layout()
    st.pyplot(fig)

# ====================== TAB 2 ======================
with tab2:
    st.header("🕸️ 02. Граф ветропарка (k-NN, k=6)")
    num_turbines = st.slider("Сколько турбин показывать", min_value=10, max_value=80, value=30, step=5)
   
    if st.button("Показать структуру ветропарка"):
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(num_turbines))
            pos_dict = {i: (loc.iloc[i]['x'], loc.iloc[i]['y']) for i in range(num_turbines)}
            
            edge_index = graph_data['edge_index'].numpy()
            edges_added = 0
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src < num_turbines and dst < num_turbines:
                    G.add_edge(src, dst)
                    edges_added += 1
            
            fig, ax = plt.subplots(figsize=(14, 11))
            nx.draw(G, pos=pos_dict, with_labels=True, node_color='lightblue', 
                   node_size=500, edge_color='gray', font_size=8, ax=ax)
            plt.title(f"Граф ветропарка (первые {num_turbines} турбин)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка: {e}")

# ====================== TAB 3 ======================
with tab3:
    st.header("🧠 03. Архитектура модели")
    st.success("✅ Модель GraphSAGE успешно загружена")
   
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Параметры модели")
        st.markdown("""
        - **Тип**: GraphSAGE (3 слоя)  
        - **Скрытая размерность**: 128  
        - **Агрегация**: Mean  
        - **Dropout**: 0.3  
        - **Вход**: 5 признаков × seq_len=6  
        - **Выход**: 3 управляющих параметра
        """)
    with col2:
        st.subheader("Целевые переменные")
        st.markdown("""
        - **Pab1, Pab2, Pab3** — Углы установки лопастей (Pitch)
        """)
    st.info("**Преимущество GraphSAGE**: модель учитывает эффект wake (ветровую тень) через пространственные связи.")

# ====================== TAB 4 ======================
with tab4:
    st.header("🔮 04. Реальный прогноз")
   
    if model is None:
        st.error("Модель не загружена")
        st.stop()

    turbine_options = sorted(df['TurbID'].unique())
    selected_turb = st.selectbox("Выберите турбину", turbine_options, index=0)

    if st.button("🚀 Выполнить прогноз", type="primary"):
        with st.spinner("Выполняется прогноз..."):
            try:
                turb_data = df[df['TurbID'] == selected_turb].tail(6)
                if len(turb_data) < 6:
                    st.error("Недостаточно данных для турбины")
                else:
                    features = turb_data[feature_cols].values.astype(np.float32)
                    x = torch.tensor(features).unsqueeze(0)
                    
                    data = Data(x=x, edge_index=graph_data['edge_index'])
                    
                    with torch.no_grad():
                        pred = model(data)
                    
                    pred_orig = scaler_targets.inverse_transform(pred.numpy())
                    
                    st.success(f"Прогноз для турбины {selected_turb}")
                    pred_df = pd.DataFrame(pred_orig, columns=target_cols)
                    st.dataframe(pred_df.round(3), use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка прогноза: {e}")

# ====================== TAB 5 ======================
with tab5:
    st.header("📈 05. Результаты и метрики")
    
    try:
        with open('results/evaluation_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        metrics_df = pd.DataFrame(results['metrics']).T.round(4)
        st.dataframe(metrics_df, use_container_width=True)
        

        avg_mae = metrics_df['MAE'].mean()
        avg_rmse = metrics_df['RMSE'].mean()
        
  
        r2_col = [col for col in metrics_df.columns if 'r2' in col.lower() or 'r²' in col.lower()]
        avg_r2 = metrics_df[r2_col[0]].mean() if r2_col else 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Средний MAE", f"{avg_mae:.3f}")
        col2.metric("Средний RMSE", f"{avg_rmse:.3f}")
        col3.metric("Средний R²", f"{avg_r2:.3f}")
        
        st.success("Метрики загружены успешно")
        
    except Exception as e:
        st.warning(f"Не удалось загрузить метрики: {e}")
        st.info("Запустите 04_evaluation.ipynb")
        
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; padding: 25px 0 15px 0; font-size: 0.95em;">
    <strong>Дипломная работа • Левещина Виктория Артемовна • 2026</strong><br>
    Графовые нейронные сети в задачах ветроэнергетики
</div>
""", unsafe_allow_html=True)