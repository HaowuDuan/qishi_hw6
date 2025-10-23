# Excel 智能体 - 数据分析助手

一个基于自然语言的Excel数据分析智能体，支持语音输入和实时分析。

## 功能特性

- 📊 **Excel文件预处理**: 处理复杂多级表头，自动识别表格结构
- 🗣️ **自然语言理解**: 支持中文和英文查询
- 🎤 **语音输入**: WebSocket实时语音识别
- 🤖 **AI代码生成**: 自动生成Python分析代码
- 📈 **数据可视化**: 自动生成图表和统计摘要
- 🔍 **数据追溯**: 明确显示使用的数据列和计算过程

## 技术栈

### 后端
- **Flask**: Web框架
- **Flask-SocketIO**: WebSocket支持
- **Pandas**: 数据处理
- **OpenPyXL**: Excel文件处理
- **ChatGPT API**: 自然语言处理和代码生成
- **SpeechRecognition**: 语音识别

### 前端
- **React**: 用户界面
- **Tailwind CSS**: 样式框架
- **Socket.IO**: 实时通信
- **Axios**: HTTP客户端

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置ChatGPT API

在 `app.py` 中设置你的OpenAI API密钥：

```python
CHATGPT_API_KEY = "your-openai-api-key"  # 替换为你的API密钥
```

### 3. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5001` 启动。

## 使用说明

### 1. 上传Excel文件
- 支持 `.xlsx` 和 `.xls` 格式
- 系统会自动处理合并单元格和多级表头

### 2. 输入分析需求
- **文本输入**: 在文本框中描述分析需求
- **语音输入**: 点击麦克风按钮进行语音输入

### 3. 查看分析结果
- **生成的Python代码**: 显示AI生成的分析代码
- **分析结果**: 表格、图表或统计摘要
- **数据追溯**: 显示使用的数据列和计算过程

## 示例查询

- "帮我分析各地区销售趋势"
- "按月份统计销售额"
- "比较不同产品的销售表现"
- "分析客户购买行为"
- "生成销售报告"

## 项目结构

```
├── app.py                 # Flask后端应用
├── templates/
│   └── index.html        # React前端页面
├── requirements.txt      # Python依赖
└── README.md            # 项目说明
```

## 核心功能实现

### Excel预处理
- 处理合并单元格
- AI识别表格结构
- 标准化数据格式

### 自然语言处理
- 意图识别
- 参数提取
- 代码生成

### 数据追溯
- 源文件追踪
- 使用列识别
- 计算过程记录

## 注意事项

1. 需要有效的OpenAI API密钥
2. 语音功能需要浏览器麦克风权限
3. 建议使用Chrome或Firefox浏览器
4. 大文件处理可能需要较长时间

## 故障排除

### 常见问题

1. **API密钥错误**: 检查OpenAI API密钥是否正确设置
2. **语音识别失败**: 确保浏览器有麦克风权限
3. **文件上传失败**: 检查文件格式是否为Excel格式
4. **分析超时**: 尝试使用较小的Excel文件

### 调试模式

启用Flask调试模式：

```python
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
```

## 开发说明

这是一个作业项目，实现了Excel智能体的核心功能：

- ✅ 文件预处理（复杂多级表头处理）
- ✅ 自然语言解析
- ✅ 代码生成与执行
- ✅ 数据追溯
- ✅ WebSocket实时语音输入
- ✅ 用户友好的Web界面

## 许可证

MIT License
