1. Подготовка и планирование (1 месяц)
Цели:
	Мы определили основные задачи проекта: предсказание стоимости четырёх ключевых активов (WTI, Brent, S&P 500 и Nasdaq-100).
	В качестве признаков будем использовать фичи извлеченные из текстовой информаций новостных источников, а так же данные о поведении самих временных рядов.


Действия:
- Проведён анализ существующих решений и подходов к прогнозированию на основе текстовых данных.
- Изучены методы обработки текстов и создания эмбеддингов.

Результат: Полное понимание целей проекта и разработка детального плана, включающего ключевые шаги и распределение обязанностей между членами команды.

---

2. Сбор данных (1.5 месяца)
Цели:
- Сбор текстовых данных из различных источников (API, новостные сайты, социальные сети, форумы).
- Получение исторических данных о ценах на фьючерсы для сопоставления с текстовыми данными.

Действия:
- Настроен скрип выгрузки с помощью Yahoo Finance для получения исторических данных по стоимости активов.
- Настроен скрип выгрузки новостной информации с помощью Webz.io API.
- Проверка данных на полноту и качество, создание структурированной системы хранения для текстовой и временной информации.
- Проведен разведочный анализ данных относительно временных рядов и предполагаемых признаков


Результат: Создан набор данных, который представляет из себя историческую информацию о каждом из используемых активов и текстовые данные из новостных источников по релевантным тематикам. Проведен EDA.

---

3. Предобработка данных (1.5 месяца)
Цели:
- Проведение предобработки текстовых данных для подготовки их к анализу и обучению моделей.

Действия:
- Очистка данных от шума, токенизация, лемматизация, стемминг.
- Эксперименты с разными методами предобработки для улучшения качества текстов (например, удаление стоп-слов, анализ синтаксиса).
- Сопоставление текстов с временными метками и данными о ценах на фьючерсы.

Результат: Очищенный и подготовленный набор текстовых данных с привязкой к ценам на фьючерсы.

---

4. Подготовка эмбеддингов и начальные эксперименты (2 месяца)
Цели:
- Подготовка эмбеддингов текста для дальнейшего использования в моделях.
- Начальный эксперимент с различными подходами к представлению текстов.

Действия:
- Реализация классических методов эмбеддинга (TF-IDF, Bag-of-Words).
- Эксперименты с современными эмбеддингами (Word2Vec, GloVe, BERT).
- Сравнение и оценка качества различных эмбеддингов на основе корреляции с ценами фьючерсов.

Результат: Подготовленные эмбеддинги текстов и определенный базовый подход для дальнейшего использования.

---

5. Разработка и оптимизация ML-классификаторов (1.5 месяца)
Цели:
- Построение и сравнение различных ML-моделей для предсказания движения цен фьючерсов.

Действия:
- Разработка базовых моделей (логистическая регрессия, Random Forest, XGBoost).
- Проведение гиперпараметрической оптимизации и оценка качества моделей.
- Эксперименты с разными эмбеддингами и их влиянием на качество моделей.

Результат: Обученные ML-модели с определением лучшего сочетания эмбеддингов и алгоритмов.

---

6. Разработка и исследование DL-классификаторов (2 месяца)
Цели:
- Построение и исследование глубоких нейронных сетей (DL) для предсказания движения цен на фьючерсы.

Действия:
- Реализация нейронных сетей на основе RNN, LSTM, GRU для работы с временными рядами и текстами.
- Эксперименты с трансформерами (например, BERT, GPT) для обработки текста.
- Гиперпараметрический тюнинг и сравнение DL-моделей с ML-моделями.

Результат: Тренированные DL-классификаторы и определение наиболее эффективной архитектуры для задачи.

---

7. Дополнительные эксперименты и доработка (1 месяц)
Цели:
- Улучшение качества моделей, интеграция дополнительных источников данных (например, рыночные индикаторы, объем торгов).
- Анализ влияния дополнительных факторов на точность предсказаний.

Действия:
- Проведение дополнительных экспериментов для улучшения точности моделей.
- Интеграция дополнительных данных и анализ их влияния на предсказания.

Результат: Оптимизированные модели, учитывающие дополнительные данные для более точных предсказаний.

---

8. Разработка прототипа сервиса (1.5 месяца)
Цели:
- Создание сервиса для предсказания движения цен на фьючерсы на основе введенных текстовых данных.

Действия:
- Разработка интерфейса для ввода текстов и получения результатов предсказания.
- Интеграция лучших моделей в сервис и тестирование их на реальных данных.

Результат: Рабочий прототип веб-сервиса с возможностью ввода текста и получения прогноза движения цен на фьючерсы.

---

9. Тестирование, оценка и финальная доработка (1 месяц)
Цели:
- Тестирование сервиса на устойчивость и точность, исправление ошибок и улучшение юзабилити.

Действия:
- Проведение тестирования с различными наборами данных и оценка результатов.
- Устранение ошибок, улучшение производительности и удобства использования.

Результат: Стабильный и оптимизированный веб-сервис для предсказания движения цен.

---

10. Финализация проекта и подготовка к презентации (1 месяц)
Цели:
- Подготовка детального отчета по проекту, результатов и выводов.
- Подготовка презентации и репетиция для защиты проекта.

Действия:
- Создание итогового отчета, включающего все этапы проекта, результаты и анализ.
- Подготовка и тестирование презентации, обеспечение качественного выступления.

Результат: Полный отчет о проекте и готовая презентация для защиты и демонстрации работы.
