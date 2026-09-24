import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- ДАННЫЕ --------------------
# 20 цветов (R, G, B) и их метки (1 - красный, 0 - нет)
colors = [
    # Красные оттенки (метка 1)
    (255, 0, 0),     # 1 - чистый красный
    (200, 0, 0),     # 1 - тёмно-красный
    (255, 50, 50),   # 1 - светло-красный
    (255, 100, 0),   # 1 - оранжевый (красный с жёлтым)
    (255, 150, 0),   # 1 - светло-оранжевый
    (180, 50, 50),   # 1 - бордовый
    (128, 0, 0),     # 1 - тёмно-бордовый
    (150, 0, 50),    # 1 - вишнёвый (красный с синим оттенком)
    (200, 80, 80),   # 1 - розовато-красный (всё ещё красный)
    # Не красные (метка 0)
    (255, 200, 200), # 0 - розовый
    (0, 255, 0),     # 0 - зелёный
    (0, 0, 255),     # 0 - синий
    (255, 255, 0),   # 0 - жёлтый
    (0, 255, 255),   # 0 - голубой
    (255, 0, 255),   # 0 - пурпурный (магента)
    (100, 100, 100), # 0 - серый
    (0, 0, 0),       # 0 - чёрный
    (255, 255, 255), # 0 - белый
    (100, 200, 100), # 0 - салатовый
    (50, 50, 150),   # 0 - тёмно-синий
]

# Метки: 1 для первых 10, 0 для последних 10
labels = [1]*9 + [0]*11


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def stair(x):
    return 1 if x>0 else 0

class Neuron:
    def __init__(self, weights, bias, activation='sigmoid'):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = sigmoid if activation == 'sigmoid' else tanh if activation == 'tanh' else relu if activation == 'relu' else stair

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

    def __call__(self, inputs):
        return self.forward(inputs)
            
my_neuron = Neuron(weights=[6.7, -6.7, -6.7], bias=-6.7, activation='stair') # FUCK SIX-SEVEN DAAAAMNNNNNN (на самом деле 10 -10 -10 тоже работало)

# Функция проверки
def test_neuron(neuron):
    correct = 0
    errors = []
    for i, (R, G, B) in enumerate(colors):
        pred = neuron([R, G, B])
        if pred == labels[i]:
            correct += 1
        else:
            errors.append((i, R, G, B, labels[i], pred))

    accuracy = correct / len(colors) * 100
    print(f"Точность: {accuracy:.1f}% ({correct}/{len(colors)})")

    if errors:
        print("Ошибки на цветах (индекс, (R,G,B), ожидалось, получено):")
        for e in errors:
            print(e)
    else:
        print("✅ ВЕРНО! Все цвета классифицированы правильно.")


# Проверяем
test_neuron(my_neuron)

