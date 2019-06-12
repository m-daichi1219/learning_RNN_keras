import pandas as pd
import math
import random

import sys

sys.path.append("G:\PycharmProjects\learning_RNN_keras\sin_example")
print(sys.path)

from sin_example.functions import *

# シード値
random.seed(0)

# 乱数の係数
random_factor = 0.05

# サイクル当たりのステップ数
steps_per_cycle = 80

# 生成するサイクル数
number_of_cycles = 50

df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor))
df[["sin_t"]].head(steps_per_cycle * 2).plot()
# plt.show()

length_of_sequences = 100
(x_train, y_train), (x_test, y_test) = train_test_split(df[["sin_t"]], n_prev=length_of_sequences)
print("T")