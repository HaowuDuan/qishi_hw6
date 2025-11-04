"""
Excel Intelligent Agent - Clean Architecture
Integrates file upload, preprocessing, LLM analysis, and code execution
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import json
import os
import uuid
import tempfile
import requests
import logging
from datetime import datetime
import base64
import io
import speech_recognition as sr
from pydub import AudioSegment

# Import our modules
from dismantle_excel import main_unmerge_file, get_excel_data
from execute_python import model_execute_main

# Import xAI SDK
from xai_sdk import Client
from xai_sdk.chat import user, system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'excel-agent-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# LLM Configuration - xAI
XAI_API_KEY = "can not upload to github" # 请替换为你的xAI API密钥
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Global storage for processed files
processed_files = {}

class LLMService:
    """LLM service for generating Python analysis code using xAI SDK"""
    
    def __init__(self):
        self.api_key = XAI_API_KEY
        self.client = Client(api_key=self.api_key)
    
    def generate_analysis_code(self, user_query, data_structure, data_sample):
        """Generate Python code based on user query and data structure"""
        try:
            logger.info(f"Generating code for query: {user_query}")
            logger.info(f"Data structure type: {type(data_structure)}")
            logger.info(f"Data sample type: {type(data_sample)}")
            
            prompt = self._build_analysis_prompt(user_query, data_structure, data_sample)
            response = self._call_llm(prompt)
            
            if response:
                return self._extract_code(response)
            else:
                # Fallback to simple analysis code
                logger.warning("xAI API failed, using fallback analysis code")
                return self._get_fallback_code()
                
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}", exc_info=True)
            return self._get_fallback_code()
    
    def _get_fallback_code(self):
        """Fallback analysis code when LLM fails"""
        return """
# 基础数据分析
print("=== 数据基本信息 ===")
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print("\\n=== 数据预览 ===")
print(df.head())

# 数据类型分析
print("\\n=== 数据类型 ===")
print(df.dtypes)

# 缺失值分析
print("\\n=== 缺失值统计 ===")
print(df.isnull().sum())

# 尝试转换数值型列
print("\\n=== 尝试转换数值型列 ===")
numeric_cols = []
for col in df.columns:
    if col not in ['Unnamed: 0', 'Unnamed_0']:  # 跳过索引列
        try:
            # 尝试转换为数值型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
                print(f"列 {col} 转换为数值型成功")
        except:
            print(f"列 {col} 无法转换为数值型")

print(f"\\n找到数值型列: {numeric_cols}")

if len(numeric_cols) > 0:
    print("\\n=== 数值型列统计 ===")
    print(df[numeric_cols].describe())
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 数值型列的分布图
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:4]):  # 最多显示4个数值列
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=20, alpha=0.7)
                axes[i].set_title(f'{col} 分布图')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('频次')
        # 隐藏多余的子图
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    # 2. 相关性热力图
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('数值型列相关性热力图')
        plt.tight_layout()
        plt.show()
    
    # 3. 箱线图
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12, 6))
        df[numeric_cols].boxplot()
        plt.title('数值型列箱线图')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 4. 散点图矩阵（如果列数不多）
    if len(numeric_cols) <= 4 and len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        pd.plotting.scatter_matrix(df[numeric_cols], alpha=0.6, figsize=(12, 10))
        plt.suptitle('数值型列散点图矩阵')
        plt.tight_layout()
        plt.show()
    
    # 5. 时间序列分析（如果有日期列）
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        plt.figure(figsize=(15, 8))
        for i, num_col in enumerate(numeric_cols[:2]):  # 最多显示2个数值列
            plt.subplot(1, 2, i+1)
            for date_col in date_cols[:1]:  # 只使用第一个日期列
                df_plot = df[[date_col, num_col]].dropna()
                if len(df_plot) > 0:
                    plt.plot(df_plot[date_col], df_plot[num_col], marker='o')
                    plt.title(f'{num_col} 时间趋势图')
                    plt.xlabel(date_col)
                    plt.ylabel(num_col)
                    plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
else:
    print("\\n没有找到数值型列进行分析")
"""
    
    def _build_analysis_prompt(self, query, structure, sample):
        """Build structured prompt for code generation"""
        prompt = f"""
你是一个专业的数据分析师。请根据用户的需求和数据结构，生成Python代码进行数据分析。

用户需求：{query}

数据结构信息：
{json.dumps(structure, ensure_ascii=False, indent=2)}

数据样本（前5行）：
{json.dumps(sample, ensure_ascii=False, indent=2)}

请生成Python代码，要求：
1. 使用pandas处理数据，数据已经加载为df变量
2. 生成清晰的统计分析和可视化
3. 包含数据追溯信息（使用了哪些列）
4. 输出格式化的分析结果
5. 使用matplotlib/seaborn进行可视化
6. 代码要完整可执行
7. 不要硬编码数据，使用实际的df数据
8. 处理列名中的特殊字符和空格
9. 添加中文注释说明分析步骤
10. 使用df.columns查看实际列名，不要假设列名
11. 先检查列是否存在再使用
12. 使用try-except处理可能的错误
13. 只分析数值型列，跳过文本列

**重要：必须包含多种可视化图表：**
- 至少3-5个不同的图表（柱状图、折线图、散点图、热力图、箱线图等）
- 使用plt.figure(figsize=(12, 8))设置合适的图表大小
- 添加图表标题、坐标轴标签
- 使用seaborn设置美观的样式
- 如果是时间序列数据，绘制趋势图
- 如果是分类数据，绘制分布图
- 如果是数值数据，绘制相关性热力图

**数据处理要求：**
- 首先检查数据类型，将可能的数值列转换为数值型
- 使用pd.to_numeric()转换数值列，errors='coerce'处理非数值
- 跳过明显的文本列（如电机编号）
- 确保有数值型列再进行可视化

**关键约束：**
- 数据已经加载为df变量，不要重新加载数据
- 不要假设有多个DataFrame，只使用df
- 不要创建空的DataFrame或占位符
- 所有图表必须基于实际的df数据
- 使用plt.show()显示图表
- 确保代码能处理空数据或异常情况

请只返回Python代码，不要包含任何解释文字：
"""
        return prompt
    
    def _call_llm(self, prompt):
        """Call xAI API using the xAI SDK"""
        try:
            logger.info("Calling xAI API with SDK...")
            
            # Use xAI SDK to make the call
            chat = self.client.chat.create(
                model="grok-3",
                messages=[
                    system("你是一个专业的数据分析师，擅长生成Python数据分析代码。"),
                    user(prompt)
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Get the actual response
            response = chat.sample()
            logger.info(f"xAI SDK response: {response}")
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
                logger.info(f"Generated content: {content[:200]}...")
                return content
            else:
                logger.error(f"Unexpected xAI SDK response format: {response}")
                return None
            
        except Exception as e:
            logger.error(f"xAI SDK call failed: {e}")
            return None
    
    def _extract_code(self, response):
        """Extract Python code from LLM response"""
        # Remove markdown code blocks if present
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            return response.strip()

class ExcelProcessor:
    """Excel file processing and data extraction"""
    
    def __init__(self):
        self.temp_files = []
    
    def process_excel_file(self, file_path):
        """Process Excel file and return structured data"""
        try:
            # Generate unique output path
            output_path = f"processed_{uuid.uuid4()}.xlsx"
            
            # Use dismantle_excel to preprocess
            try:
                result_path = main_unmerge_file(file_path, output_path)
                logger.info(f"Preprocessing result: {result_path}")
                
                if result_path and os.path.exists(result_path):
                    # Extract data structure and sample
                    structure = self._extract_data_structure(result_path)
                    sample = self._extract_data_sample(result_path)
                    
                    return {
                        'structure': structure,
                        'sample': sample,
                        'file_path': result_path
                    }
                else:
                    # If preprocessing fails, try to use original file
                    logger.warning("Preprocessing failed, using original file")
                    structure = self._extract_data_structure(file_path)
                    sample = self._extract_data_sample(file_path)
                    
                    return {
                        'structure': structure,
                        'sample': sample,
                        'file_path': file_path
                    }
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}", exc_info=True)
                # Fallback to original file
                logger.warning("Preprocessing failed, using original file")
                structure = self._extract_data_structure(file_path)
                sample = self._extract_data_sample(file_path)
                
                return {
                    'structure': structure,
                    'sample': sample,
                    'file_path': file_path
                }
                
        except Exception as e:
            logger.error(f"Excel processing failed: {e}", exc_info=True)
            return None
    
    def _extract_data_structure(self, file_path):
        """Extract data structure information"""
        try:
            # Read all sheets
            all_sheets = pd.read_excel(file_path, sheet_name=None)
            structure = {}
            
            for sheet_name, df in all_sheets.items():
                structure[sheet_name] = {
                    'columns': list(df.columns),
                    'shape': list(df.shape),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                    'null_counts': {col: int(count) for col, count in df.isnull().sum().to_dict().items()}
                }
            
            return structure
            
        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
            return {}
    
    def _extract_data_sample(self, file_path):
        """Extract data sample for LLM analysis"""
        try:
            # Read first sheet and get sample
            df = pd.read_excel(file_path, sheet_name=0)
            logger.info(f"Sample extraction - Shape: {df.shape}")
            logger.info(f"Sample extraction - Columns: {list(df.columns)}")
            
            # Convert to string to ensure JSON serialization
            sample = df.head(5).astype(str).to_dict('records')
            return sample
            
        except Exception as e:
            logger.error(f"Sample extraction failed: {e}")
            return []

class VoiceRecognitionService:
    """Voice recognition service for real-time speech-to-text"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # 配置语音识别参数
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
    
    def recognize_audio(self, audio_data):
        """Convert audio data to text using speech recognition"""
        try:
            logger.info("Starting voice recognition...")
            
            # 将base64音频数据转换为音频对象
            audio_bytes = base64.b64decode(audio_data)
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            
            # 转换为speech_recognition可识别的格式
            audio_source = sr.AudioData(
                audio_segment.raw_data,
                audio_segment.frame_rate,
                audio_segment.sample_width
            )
            
            # 使用Google语音识别API（免费）
            try:
                text = self.recognizer.recognize_google(audio_source, language='zh-CN')
                logger.info(f"Voice recognition result: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                return "抱歉，无法识别语音内容，请重试"
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                return "语音识别服务暂时不可用，请稍后重试"
                
        except Exception as e:
            logger.error(f"Voice recognition failed: {e}", exc_info=True)
            return f"语音识别出错: {str(e)}"

# Initialize services
llm_service = LLMService()
excel_processor = ExcelProcessor()
voice_service = VoiceRecognitionService()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = f"upload_{file_id}_{file.filename}"
        file_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(file_path)
        
        # Process Excel file
        logger.info(f"Processing file: {file_path}")
        processed_data = excel_processor.process_excel_file(file_path)
        logger.info(f"Processed data result: {processed_data is not None}")
        
        if processed_data:
            # Store processed data
            processed_files[file_id] = {
                'file_path': processed_data['file_path'],
                'structure': processed_data['structure'],
                'sample': processed_data['sample'],
                'upload_time': datetime.now().isoformat()
            }
            
            # Clean up original file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'message': 'File processed successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'File processing failed'})
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/check_preprocessing', methods=['POST'])
def check_preprocessing():
    """Check preprocessing status"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        logger.info(f"Checking preprocessing for file_id: {file_id}")
        logger.info(f"Available files: {list(processed_files.keys())}")
        
        if file_id in processed_files:
            file_data = processed_files[file_id]
            logger.info(f"File data: {file_data}")
            
            # Get data preview
            df = pd.read_excel(file_data['file_path'], sheet_name=0)
            preview_data = df.head(10).to_dict('records')
            columns = list(df.columns)
            
            logger.info(f"Data preview generated, columns: {columns}")
            
            return jsonify({
                'success': True,
                'status': 'completed',
                'data_preview': preview_data,
                'columns': columns
            })
        else:
            logger.error(f"File not found: {file_id}")
            return jsonify({'success': False, 'error': 'File not found'})
            
    except Exception as e:
        logger.error(f"Check preprocessing error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Perform data analysis"""
    try:
        data = request.get_json()
        query = data.get('query')
        file_id = data.get('file_id')
        
        logger.info(f"Analysis request: query='{query}', file_id='{file_id}'")
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        if file_id not in processed_files:
            logger.error(f"File not found: {file_id}")
            return jsonify({'success': False, 'error': 'File not found'})
        
        file_data = processed_files[file_id]
        
        # Generate analysis code using LLM
        generated_code = llm_service.generate_analysis_code(
            query, 
            file_data['structure'], 
            file_data['sample']
        )
        
        if not generated_code:
            return jsonify({'success': False, 'error': 'Failed to generate analysis code'})
        
        # Prepare data for code execution
        code_with_data = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_excel('{file_data['file_path']}')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib后端，确保图表能正确显示
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 数据追溯信息
print("=== 数据追溯信息 ===")
print(f"数据形状: {{df.shape}}")
print(f"使用的列: {{list(df.columns)}}")
print("\\n=== 数据预览 ===")
print(df.head())

# 开始分析
{generated_code}

# 保存所有图表到文件
import os
import uuid
chart_dir = "charts"
if not os.path.exists(chart_dir):
    os.makedirs(chart_dir)

# 获取当前所有图形并保存
figs = [plt.figure(i) for i in plt.get_fignums()]
for i, fig in enumerate(figs):
    chart_path = f"{{chart_dir}}/chart_{{i+1}}_{{uuid.uuid4().hex[:8]}}.png"
    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"图表 {{i+1}} 已保存到: {{chart_path}}")

print(f"\\n总共生成了 {{len(figs)}} 个图表")
"""
        
        # Execute the code
        result = model_execute_main(code_with_data)
        
        return jsonify({
            'success': True,
            'generated_code': generated_code,
            'result': result,
            'columns_used': _extract_columns_used(generated_code, file_data['structure'])
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def _extract_columns_used(code, structure):
    """Extract which columns were used in the analysis"""
    try:
        used_columns = []
        for sheet_name, sheet_info in structure.items():
            for col in sheet_info['columns']:
                if col in code:
                    used_columns.append(f"{sheet_name}.{col}")
        return used_columns
    except:
        return []

@socketio.on('voice_input')
def handle_voice_input(data):
    """Handle real-time voice input and convert to text"""
    try:
        logger.info("Received voice input via WebSocket")
        
        # 检查是否包含音频数据
        if 'audio' not in data:
            emit('voice_result', {'text': '未收到音频数据，请重试'})
            return
        
        # 获取音频数据
        audio_data = data['audio']
        logger.info(f"Audio data length: {len(audio_data)}")
        
        # 使用语音识别服务转换音频为文本
        recognized_text = voice_service.recognize_audio(audio_data)
        
        # 发送识别结果
        emit('voice_result', {'text': recognized_text})
        logger.info(f"Voice recognition completed: {recognized_text}")
        
    except Exception as e:
        logger.error(f"Voice input handling failed: {e}", exc_info=True)
        emit('error', {'message': f'语音处理出错: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
