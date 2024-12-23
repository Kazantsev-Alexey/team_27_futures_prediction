# Функционал сервиса в соответствии с пунктами задания по Streamlit

## 1. Загрузка датасета, совпадающего по формату с датасетом используемом в вашем проекте

В нашем случае, так как у нас уже есть сохраненные данные на хосте, то нам нужна выгрузка из базы данных и альтернатива — это загрузка из CSV. Метод должен включать в себя возможность выгрузить имеющиеся данные либо загрузить новые.

Для тикеров, которые у нас уже обучены, нам необходимо просто обратиться к базе данных и соединить две таблички. Можно, наверное, это делать и целиком на SQL, но как вариант у меня есть готовый пример, который выглядит вот так:

```
tickers = ['CL=F', 'BZ=F', 'SPY', 'QQQ']

finbert_query = """
    SELECT * FROM daily_finbert_sentiment
"""
feature_query = """
    SELECT * FROM feature_data
"""

# min&max dates 
start_date = pd.to_datetime(sentiment.business_date.min())
end_date = pd.to_datetime(sentiment.business_date.max()) 


all_dates = pd.date_range(start=start_date, end=end_date)
all_combinations = pd.MultiIndex.from_product([tickers, all_dates], names=['ticker', 'business_date']).to_frame(index=False)
all_combinations = all_combinations.merge(sentiment,'left',['business_date','ticker']).merge(features,'left',['business_date','ticker','asset_name'])
# all_combinations = all_combinations.merge(sentiment,'left',['business_date','ticker']).merge(features[['business_date','ticker','close']],'left',['business_date','ticker'])

all_combinations['weekday'] = pd.to_datetime(all_combinations.business_date).dt.weekday
all_combinations['month'] = pd.to_datetime(all_combinations.business_date).dt.month
```

Здесь all_combinations – это просто все наши признаки, которые мы собрали ранее по тех индикаторам и сентиментам.

Во втором варианте с загрузкой новых данных, метод должен принимать CSV, проверять его на соответствии со столбцами из all_combinations, и после этого сохранить его в такой же датафрейм.


## 2. Демонстрация аналитики / EDA – 2 балла

В этом задании нам по большому счёту уже нечего делать, потому что весь EDA проведён нами давно и довольно подробно. Практически все признаки, которые мы отобрали, сгенерированы из временного ряда цены и сентимента, поэтому проводить на них какой-то разведочный анализ уже неэффективно. Но чтобы нам засчитали чекпоинт, мы можем сделать следующее. После загрузки датасета:

- Вывести описательные статистики
- Заполнить пропуски если есть
- Показать корреляцию признаков.
- Для нескольких признаков с наибольшей корреляцией показать scatter plots относительно целевой переменной
- Построить pair plots (я, кстати, сам на нашем датасете этого не делал, так что не уверен, что это будет полезно, но почему бы и нет, если хватит времени)


## 3. Создание новой модели и выбор гиперпараметров

Так как у нас не самый маленький набор признаков в изначальном датасете, то мы не можем просто обучать модель, выкидывая по одному признаку, потом по два, и так далее,
потому что это никогда не обучится. Самым оптимальным способом будет обучать ElasticNet с перебором гиперпараметров.
Как только будет обучена лучшая модель, посмотрим, какие признаки и какие веса остались, именно их и оставляем, и после этого используем Best Model из GridSearch и не обнуленные признаки.

---

## 4. Просмотр информации о модели и полученных кривых обучения

Мы не используем бустинг, поэтому кривые обучения посмотреть мы не можем, но мы можем посмотреть метрики, которые эти модели выдают,
 и мы можем отрисовать графики прогнозов, которые в конечном счете получились. В идеальном варианте модель уже сделала отбор признаков на кросс валидации,
поэтому нам не нужна валидационная выборка из датасета, делим его только на трейн и тест. 

С прошлого чекпоинта у нас осталась функция, которая обучает Ridge и сразу выводит анализ остатков и график прогноза.
В принципе можем отталкиваться от неё, но надо немного переработать чтобы сама модель, скейлер и признаки сохранялись в БД, это нужно для следующего пункта.
Если перенести все графики на plotly, то получится выполнить задание из бонусной части про интерактивные графики.


```
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from scipy import stats

def run_model(target,ticker_name, test_len, to_delete, model, shift_days, all_combinations):
    one_asset = all_combinations.loc[all_combinations.ticker == ticker_name]
    actual_data = one_asset.ffill()

    # Сдвигаем next_day_close на shift_days вперёд 
    actual_data['next_day_close'] = actual_data['close'].shift(-shift_days)
    columns_to_shift = [col for col in actual_data.columns if col not in ['next_day_close', 'business_date']]

    # Расчёт изменения цены
    actual_data['price_change'] = actual_data['next_day_close'] - actual_data['close']
    actual_data = actual_data.dropna().reset_index(drop=True)

    # Формирование обучающей и тестовой выборки
    X_train = actual_data.drop(to_delete, axis=1, errors='ignore').iloc[:-test_len].select_dtypes(['int', 'float'])
    y_train = actual_data[target].iloc[:-test_len]
    X_test = actual_data.drop(to_delete, axis=1, errors='ignore').iloc[-test_len:].select_dtypes(['int', 'float'])
    y_test = actual_data[target].iloc[-test_len:]
    chosen_features = X_train.columns
    
    dates_train = actual_data.business_date.iloc[:-test_len]
    dates_test = actual_data.business_date.iloc[-test_len:]

    # Добавляем стандарт скейлер
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели
    model.fit(X_train_scaled, y_train)
    best_model = model

    # Предсказания
    y_train_pred = best_model.predict(X_train_scaled) 
    y_test_pred = best_model.predict(X_test_scaled) 

    # Метрики
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    print(f"Train MAPE: {train_mape:.4f}")
    print(f"Test MAPE: {test_mape:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Вызываем функцию для отрисовки графиков
    plot_all_graphs(
        target, ticker_name, test_len, actual_data, y_train, y_train_pred, y_test, y_test_pred,
        dates_train, dates_test, best_model, X_train, scaler
    )

    return best_model, scaler, chosen_features


def plot_all_graphs(
    target, ticker_name, test_len, actual_data, y_train, y_train_pred, y_test, y_test_pred,
    dates_train, dates_test, best_model, X_train, scaler
):
    # График целевой переменной 
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train.values, label='Actual train change', color='blue', linestyle='dotted')
    plt.plot(dates_train, y_train_pred, label='Predicted train change', color='blue')
    plt.plot(dates_test, y_test.values, label='Actual test change', color='orange', linestyle='dotted')
    plt.plot(dates_test, y_test_pred, label='Predicted test change', color='orange')

    plt.title(f"Actual vs predicted {target} values for {ticker_name}")
    plt.xlabel("Date")
    plt.ylabel("Price change")
    plt.legend(title="Legend")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.show()

    if target == 'price_change':
        # График цен следующего дня
        # Фактическая и предсказанная цена следующего дня
        actual_next_day_price_train = actual_data['next_day_close'].iloc[:-test_len].values
        predicted_next_day_price_train = actual_data['close'].iloc[:-test_len].values + y_train_pred
        actual_next_day_price_test = actual_data['next_day_close'].iloc[-test_len:].values
        predicted_next_day_price_test = actual_data['close'].iloc[-test_len:].values + y_test_pred
        
        
        restored_train_mse = mean_squared_error(actual_next_day_price_train, predicted_next_day_price_train)
        restored_test_mse = mean_squared_error(actual_next_day_price_test, predicted_next_day_price_test)
        restored_train_mape = mean_absolute_percentage_error(actual_next_day_price_train, predicted_next_day_price_train)
        restored_test_mape = mean_absolute_percentage_error(actual_next_day_price_test, predicted_next_day_price_test)
        
        print(f"Restored train MAPE: {restored_train_mape:.4f}")
        print(f"Restored test MAPE: {restored_test_mape:.4f}")
        print(f"Restored train MSE: {restored_train_mse:.4f}")
        print(f"Restored test MSE: {restored_test_mse:.4f}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates_train, actual_next_day_price_train, label='Actual price (train)', color='green', linestyle='dotted')
        plt.plot(dates_train, predicted_next_day_price_train, label='Predicted price (train)', color='green')
        plt.plot(dates_test, actual_next_day_price_test, label='Actual price (test)', color='red', linestyle='dotted')
        plt.plot(dates_test, predicted_next_day_price_test, label='Predicted price (test)', color='red')

        plt.title("Actual vs predicted restored price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(title="Legend")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.show()

    # Остатки
    residual_train = y_train - y_train_pred
    residual_test = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(residual_train, bins=30, kde=True, ax=axes[0,0], color='blue')
    axes[0,0].set_title('Train Residuals Distribution')

    sns.histplot(residual_test, bins=30, kde=True, ax=axes[1,0], color='orange')
    axes[1,0].set_title('Test Residuals Distribution')

    stats.probplot(residual_train, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Train Residuals Q-Q Plot')

    stats.probplot(residual_test, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Test Residuals Q-Q Plot')

    plt.tight_layout()
    plt.show()

    shapiro_train = stats.shapiro(residual_train)
    shapiro_test = stats.shapiro(residual_test)

    print(f"Shapiro-Wilk Test for Train Residuals: Statistic={shapiro_train.statistic:.4f}, p-value={shapiro_train.pvalue:.4f}")
    print(f"Shapiro-Wilk Test for Test Residuals: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

    # Важность признаков 
    if hasattr(best_model, 'coef_'):
        feature_importances = best_model.coef_
        feature_names = X_train.columns

        # Сортируем признаки по значению коэффициента
        sorted_idx = np.argsort(np.abs(feature_importances))[::-1]
        sorted_features = feature_names[sorted_idx]
        sorted_importances = feature_importances[sorted_idx]

        plt.figure(figsize=(10, 10))
        plt.barh(sorted_features, sorted_importances)
        plt.title("Feature Importances (Model Coefficients)")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

```

## 5. Инференс с использованием ранее обученной модели – 1 балл

У нас уже есть несколько обученных моделей, которые хранятся вот здесь https://github.com/Maksim-de/team_27_futures_prediction/tree/main/saved_models. 
Но в рамках этого пункта надо бы реализовать их выгрузку именно из базы данных, а не из репозитория. Понадобится новая функция для выгрузки моделей и предикта, которая сохраняет предсказание и параметры для вызова plot_all_graphs

## 6. В приложении должно быть реализовано полноценное логирование с механизмом ротации. Логи складываем в папку logs\
Тут ничего не подскажу, 0 опыта в логировании. 


