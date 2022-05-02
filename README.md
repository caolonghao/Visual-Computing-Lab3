# Mesh Processing

## 文件列表

* base_model.py 定义mesh模型，尽量不要修改提供的函数
* /models 提供的mesh文件，可供测试
* simplify.py mesh简化部分
* smooth.py mesh平滑部分
* subdivision.py mesh细分部分
* vis_mesh.py 用于mesh可视化

## 环境配置

```
pip install numpy
pip install pyvista
```

## 可视化结果

用以下命令可视化mesh查看效果

```
python vis_mesh.py -i user-mesh-path
```

## 任务要求

本lab分成三个部分：曲面简化(6'), 曲面平滑(3'), 曲面细分(3')，满分12分。

完成`simplify.py`, `smooth.py`, `subdivision.py`三个部分的代码填空，每个部分具体提示和要求参见代码中的注释。
results文件夹中的样例结果可供参考，完成报告时**不得**再使用`dinosaur.obj`作为效果展示。

最终报告提交到教学网上。
