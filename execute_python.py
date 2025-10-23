# pip install jupyter_client
# pip install ipykernel
from jupyter_client.manager import start_new_kernel
import logging

logger = logging.getLogger(f'2brain.{__name__}')


def run_code(code, client):
    """执行代码并获取输出结果"""
    try:
        # 发送代码执行请求
        client.execute(code)

        # 设置超时时间
        TIMEOUT = 30

        # 获取输出结果
        output = []
        error_output = []
        display_data = []
        
        while True:
            try:
                msg = client.get_iopub_msg(timeout=TIMEOUT)
                msg_type = msg['header']['msg_type']
                content = msg['content']

                if msg_type == 'stream':
                    output.append(content['text'])
                elif msg_type == 'execute_result':
                    if 'text/plain' in content.get('data', {}):
                        output.append(content['data']['text/plain'])
                elif msg_type == 'display_data':
                    # 处理图表显示数据
                    if 'image/png' in content.get('data', {}):
                        import base64
                        img_data = content['data']['image/png']
                        # 将base64图片数据转换为可显示的格式
                        display_data.append(f"[图表已生成 - 图片数据长度: {len(img_data)}]")
                    elif 'text/plain' in content.get('data', {}):
                        display_data.append(content['data']['text/plain'])
                elif msg_type == 'error':
                    error_traceback = '\n'.join(content['traceback'])
                    error_output.append(f"ERROR:\n{error_traceback}")
                    # 不要立即返回，继续收集其他输出
                elif msg_type == 'status' and content['execution_state'] == 'idle':
                    break

            except Exception as e:
                logging.error(f"获取输出时发生错误: {str(e)}")
                break

        # 如果有错误，优先返回错误信息
        if error_output:
            return '\n'.join(error_output)
        
        # 合并所有输出
        all_output = output + display_data
        return '\n'.join(all_output) if all_output else "No output"

    except Exception as e:
        logging.error(f"代码执行发生错误: {str(e)}")
        return f"Exception occurred: {str(e)}"


def model_execute_main(command):
    """主函数:创建内核、执行代码并清理资源"""
    kernel_manager = None
    client = None

    try:
        # 创建新内核
        kernel_manager, client = start_new_kernel()
        logging.info(f"正在执行代码: {command}")

        # 执行代码并获取结果
        result = run_code(command, client)
        return result

    except Exception as e:
        logging.error(f"执行过程发生错误: {str(e)}")
        return f"Execution failed: {str(e)}"

    finally:
        # 清理资源
        if client:
            try:
                client.stop_channels()
            except Exception as e:
                logging.error(f"停止通道时发生错误: {str(e)}")

        if kernel_manager:
            try:
                kernel_manager.shutdown_kernel()
            except Exception as e:
                logging.error(f"关闭内核时发生错误: {str(e)}")

        del client
        del kernel_manager


if __name__ == "__main__":
    # 测试代码
    test_code = '''
import math
print("pi =", round(math.pi, 2))
'''
    res = model_execute_main(test_code)
    print(res)
