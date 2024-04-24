import numpy as np

print(
    "cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat"
)

user_in_str = input(
    "请输入蘑菇的各个特征值(按照以上特征的顺序输入，并用','分隔各个特征的值)："
)

X_te = np.array(user_in_str.split(","))
