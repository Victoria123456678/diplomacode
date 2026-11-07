# Графовые нейронные сети в задачах ветроэнергетики

**Выпускная квалификационная работа бакалавра**  
Московский авиационный институт (МАИ)  
Автор: Левещина Виктория Артемовна  
Год: 2026

---

## О проекте

Репозиторий содержит код дипломной работы, посвящённой применению **графовых нейронных сетей** для прогнозирования управляющих параметров ветроэнергетических установок.

**Задача:** предсказывать углы установки лопастей (Pab1, Pab2, Pab3) с учётом:
- временной динамики показателей турбин
- пространственных взаимодействий между турбинами (эффект ветровой тени — wake effect)

В качестве архитектуры выбрана **GraphSAGE**.

## Основные результаты

| Метрика | Значение |
|---------|----------|
| MAE     | 15.76    |
| RMSE    | 28.25    |
| R²      | 0.60     |

- Датасет: **SDWPF** (134 турбины, 245 дней, ~4.7 млн записей)
- Построен k-NN граф ветропарка (k=6) на основе географических координат
- Модель: GraphSAGE (3 слоя, hidden dim = 128)
- Разработано интерактивное веб-приложение на **Streamlit**

## Структура репозитория
```text
diplomacode/
├── data/                          # данные (координаты турбин)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # разведочный анализ
│   ├── 02_graph_building.ipynb    # построение графа
│   ├── 03_model_training.ipynb    # обучение GraphSAGE
│   └── 04_evaluation.ipynb        # оценка качества
├── app.py                         # Streamlit-приложение
└── README.md
```
## Технологии

- Python, PyTorch, PyTorch Geometric
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Как запустить

### 1. Клонировать репозиторий
git clone https://github.com/Victoria123456678/diplomacode.git
cd diplomacode
### 2. Установить зависимости
pip install -r requirements.txt
### 3. Запустить Streamlit-приложение
streamlit run app.py
### Примечание: 
для полной работы приложения нужны предобученные веса модели и полный датасет. Они не включены в репозиторий из-за большого размера. Полный пайплайн обучения находится в ноутбуках.
### Контакты

- Telegram: @VictoriaLeve
- Email: leveshchinavictoria@gmail.com
- Город: Москва

Открыта к стажировкам и интересным задачам.
### Скриншоты приложения
<img width="966" height="426" alt="image" src="https://github.com/user-attachments/assets/3ecd63ed-be72-4c79-a084-c55c98ca1f4f" />
<img width="973" height="444" alt="image" src="https://github.com/user-attachments/assets/d0d13e6e-6a34-4814-979b-64dd97c00060" />
<img width="973" height="430" alt="image" src="https://github.com/user-attachments/assets/1bddcdee-8fa6-4ba7-bf73-1540aa0fad68" />
<img width="973" height="372" alt="image" src="https://github.com/user-attachments/assets/54218b4a-ac10-4679-9f78-e72204d36b56" />
<img width="973" height="388" alt="image" src="https://github.com/user-attachments/assets/df8d57ef-aef1-4a0a-b611-1024b0d5aec7" />
<img width="973" height="360" alt="image" src="https://github.com/user-attachments/assets/075905dd-1b2c-4665-9ef7-f4c2e787b2df" />

