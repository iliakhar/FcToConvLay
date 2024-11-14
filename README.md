## Цель:
Выразить FC слой через nn.Conv2d

Слой:  my_fc_layer = nn.Linear(3 * 12 * 12, 7)

## Решение

Задать следующие параметры для сверточного слоя: nn.Conv2d(3, 7, kernel_size=12, stride=1)

Представить вход к conv слою как тензор (3, 12, 12).

Изменить размерность весов. Веса соединений входных значений и нейрона слоя который преобразуется, необходимо привести к размерности 3x12x12. Провести эту операцию для каждого нейрона в слою который надо преобразовать.

## Результат

Сеть без сверточных слоев:

![convnet](https://github.com/iliakhar/FcToConvLay/blob/master/netron_res/fc_model.onnx.png)

Сеть с одним сверточным слоем:

![full_convnet](https://github.com/iliakhar/FcToConvLay/blob/master/netron_res/conv_model.onnx.png)

Обученные модели лежат в папке - model;

Изображения с визуализацией нейронных сетей - в папке netron_res.

## Запуск
### Установка зависимостей
Перейти в корневую папку и запусть:

pip install -r requirements.txt

### Доступные команды

*python main.py new_inputs* - сгенерировать новый случайный вход для сетей;

*python main.py compare_models* - вывести результаты обеих моелей;

*python main.py change_weights* - изменить веса моделей.
