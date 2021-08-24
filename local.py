# Импортируем библиотеки
import pickle

# Импортируем все библиотеки, которые необходимы для модели
import numpy as np
from sklearn.linear_model import LogisticRegression

# Загружаем сохраненную модель мз текущей папки
with open('./model.pkl', 'rb') as model_pkl:
   lr_model= pickle.load(model_pkl)


# Неизвестные данные (создаем новое наблюдение для тестирования)
# пример тестового наблюдения 1, 0, 0, 1, 1, 12, 1, 27, 100, 6, 6, 1.05, 0.95
test = np.array([[1, 0, 0, 1, 1, 12, 1, 27, 100, 6, 6, 1.05, 0.95]])
prediction = lr_model.predict(test)

# Выводим результаты на консоль
print('Predicted result for observation ' + str(test) + ' is: ' + str(prediction))


