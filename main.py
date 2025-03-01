import re
import os
import random
import logging
import json
import base64
import asyncio
import aiohttp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import *
from openai.types.chat.chat_completion import ChatCompletion
from astrbot.api.all import *
import time

# 用于跟踪每个用户的状态，防止超时或重复请求
USER_STATES = {}

@register("mccloud_meme_sender", "MC云-小馒头", "识别AI回复中的表情并发送对应表情包", "2.0")
class MemeSender(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self.found_emotions = []  # 存储找到的表情
        
        # 设置日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # 设置默认路径
        self.meme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes")
        
        # 从配置中获取路径和API密钥
        self.meme_path = self.config.get("meme_path", self.meme_path)
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "deepseek-vl2")

        # 检查配置文件是否存在
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as config_file:
                emotion_config = json.load(config_file)
                self.emotion_map = emotion_config.get("emotion_map", {})
                self.logger.info(f"表情配置文件已加载: {config_file_path}")
        else:
            # 创建表情配置文件
            self.emotion_map = {
                "生气": "angry",
                "开心": "happy",
                "悲伤": "sad",
                "惊讶": "surprised",
                "疑惑": "confused",
                "色色": "color",
                "色": "color",
                "死机": "cpu",
                "笨蛋": "fool",
                "给钱": "givemoney",
                "喜欢": "like",
                "看": "see",
                "害羞": "shy",
                "下班": "work",
                "剪刀": "scissors",
                "不回我": "reply",
                "喵": "meow",
                "八嘎": "baka",
                "早": "morning",
                "睡觉": "sleep",
                "唉": "sigh",
            }
            with open(config_file_path, 'w') as config_file:
                json.dump({"emotion_map": self.emotion_map}, config_file, ensure_ascii=False, indent=4)
            self.logger.info(f"表情配置文件已创建: {config_file_path}")

        # 检查表情包目录
        self._check_meme_directories()
    
    def _check_meme_directories(self):
        """检查表情包目录是否存在并且包含图片"""
        self.logger.info(f"表情包根目录: {self.meme_path}")
        if not os.path.exists(self.meme_path):
            self.logger.error(f"表情包根目录不存在: {self.meme_path}")
            return
            
        for emotion in self.emotion_map.values():
            emotion_path = os.path.join(self.meme_path, emotion)
            if not os.path.exists(emotion_path):
                self.logger.error(f"表情目录不存在: {emotion_path}")
                continue
                
            memes = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.gif'))]
            if not memes:
                self.logger.error(f"表情目录为空: {emotion_path}")
            else:
                self.logger.info(f"表情目录 {emotion} 包含 {len(memes)} 个图片")

    def _create_config_file(self):
        """创建配置文件并写入默认配置"""
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_conf_schema.json')
        
        with open(config_file_path, 'w') as config_file:
            json.dump(self.config, config_file, ensure_ascii=False, indent=4)
        self.logger.info(f"配置文件已创建: {config_file_path}")

    @filter.on_llm_response(priority=90)
    async def resp(self, event: AstrMessageEvent, response: LLMResponse):
        """处理 LLM 响应，识别表情"""
        if not response or not response.completion_text:
            return
        
        text = response.completion_text
        self.found_emotions = []  # 重置表情列表
        
        # 定义表情正则模式
        patterns = [
            r'\[([^\]]+)\]',  # [生气]
            r'\(([^)]+)\)',   # (生气)
            r'（([^）]+)）'    # （生气）
        ]
        
        clean_text = text
        
        # 查找所有表情标记
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                emotion = match.group(1)
                if emotion in self.emotion_map:
                    self.found_emotions.append(emotion)
                    clean_text = clean_text.replace(match.group(0), '')
        
        if self.found_emotions:
            # 更新回复文本(移除表情标记)
            response.completion_text = clean_text.strip()

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """在消息发送前处理表情"""
        if not self.found_emotions:
            return
            
        result = event.get_result()
        if not result:
            return
            
        try:
            # 创建新的消息链
            chains = []
            
            # 添加原始文本消息链
            original_chain = result.chain
            if original_chain:
                if isinstance(original_chain, str):
                    chains.append(Plain(original_chain))
                elif isinstance(original_chain, MessageChain):
                    chains.extend(original_chain)
                elif isinstance(original_chain, list):
                    chains.extend(original_chain)
                else:
                    self.logger.warning(f"未知的消息链类型: {type(original_chain)}")
            
            # 添加表情包
            for emotion in self.found_emotions:
                emotion_en = self.emotion_map.get(emotion)
                if not emotion_en:
                    continue
                    
                emotion_path = os.path.join(self.meme_path, emotion_en)
                if os.path.exists(emotion_path):
                    memes = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.gif'))]
                    if memes:
                        meme = random.choice(memes)
                        meme_file = os.path.join(emotion_path, meme)
                        
                        # 使用正确的方式添加图片到消息链
                        chains.append(Image.fromFileSystem(meme_file))
            
            # 使用 make_result() 构建结果
            result = event.make_result()
            for component in chains:
                if isinstance(component, Plain):
                    result = result.message(component.text)
                elif isinstance(component, Image):
                    result = result.file_image(component.path)
            
            # 设置结果
            event.set_result(result)
            
        except Exception as e:
            self.logger.error(f"处理表情失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        # 清空表情列表
        self.found_emotions = []

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """消息发送后的清理工作"""
        self.found_emotions = []  # 确保清空表情列表

    @filter.command("save")
    async def start_importing_images(self, event: AstrMessageEvent):
        """开始导入图片"""
        user_id = event.get_sender_id()
        USER_STATES[user_id] = {"importing": True}
        yield event.plain_result("开始导入图片，请发送你要识别的图片。输入 /esc 结束导入。")

    @filter.command("esc")
    async def end_importing_images(self, event: AstrMessageEvent):
        """结束导入图片"""
        user_id = event.get_sender_id()
        if user_id in USER_STATES:
            del USER_STATES[user_id]
            yield event.plain_result("结束导入图片。")
        else:
            yield event.plain_result("你还没有开始导入图片。")

    @event_message_type(EventMessageType.ALL)
    async def handle_image(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if user_id not in USER_STATES or not USER_STATES[user_id].get("importing"):
            return  # 如果用户没有发起请求，跳过
        
        # 检查消息中是否包含图片
        images = [c for c in event.message_obj.message if isinstance(c, Image)]
        if not images:
            return
        
        # 获取图片 URL
        image_urls = [images[i].url for i in range(len(images))]  # 获取所有图片的 URL
        response = await self.call_baidu_ai(image_urls)
        
        # 处理 AI 返回的结果
        chain = [
            Plain(f"以下是这张图片的描述：{response}")
        ]
        yield event.chain_result(chain)

    def _get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    def _get_signature(self, secret_key, timestamp):
        """计算签名"""
        import hmac
        import hashlib
        import base64
        
        # 构建规范化请求字符串
        http_method = "POST"
        path = "/v1/BCE-BEARER/token"
        params = f"expireInSeconds=86400"
        
        # 按照规范组织签名字符串
        sign_key_info = f"bce-auth-v1/{self.config['api_key']}/{timestamp}/1800"
        sign_key = hmac.new(
            secret_key.encode('utf-8'),
            sign_key_info.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        string_to_sign = f"{http_method}\n{path}\n{params}\n"  # 注意最后的换行符
        
        # 计算最终签名
        signature = hmac.new(
            sign_key,
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')

    async def get_access_token(self):
        """获取百度 AI 的 access token"""
        api_key = self.config.get("api_key", "")
        secret_key = self.config.get("secret_key", "")
        
        # 如果 API Key 以 bce-v3/ 开头，提取 ALTAK 部分作为 client_id
        if api_key.startswith('bce-v3/'):
            parts = api_key.split('/')
            if len(parts) >= 3:
                client_id = parts[1]  # 使用 ALTAK-xxx 部分作为 client_id
                self.logger.debug(f"从 bce-v3/ 格式中提取 client_id: {client_id}")
                api_key = client_id
        
        if not api_key or not secret_key:
            raise ValueError("API Key 或 Secret Key 未配置。请在配置文件中设置。")

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            'grant_type': 'client_credentials',
            'client_id': api_key,
            'client_secret': secret_key
        }

        self.logger.debug(f"正在获取 access token")
        self.logger.debug(f"请求 URL: {url}")
        self.logger.debug(f"client_id: {api_key}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as resp:
                    error_data = await resp.text()
                    self.logger.debug(f"响应状态码: {resp.status}")
                    self.logger.debug(f"响应内容: {error_data}")
                    
                    if resp.status != 200:
                        raise Exception(f"获取 access token 失败，状态码: {resp.status}, 响应: {error_data}")
                    
                    data = await resp.json()
                    if 'access_token' not in data:
                        raise Exception("获取 access token 失败，响应中没有 access_token")
                    
                    self.logger.info("成功获取 access token")
                    return data['access_token']
        except Exception as e:
            self.logger.error(f"获取 access token 失败: {str(e)}")
            raise

    async def _save_image(self, img_url, emotion):
        """保存图片到对应的情绪目录"""
        emotion_en = self.emotion_map.get(emotion)
        if not emotion_en:
            return f"未找到对应的情绪目录：{emotion}"

        # 确保目录存在
        emotion_path = os.path.join(self.meme_path, emotion_en)
        os.makedirs(emotion_path, exist_ok=True)

        try:
            # 确保 img_url 是有效的，并且以 https:// 开头
            if not img_url.startswith("https://"):
                img_url = "https://" + img_url  # 添加 https:// 前缀

            self.logger.debug(f"准备下载图片，URL: {img_url}")
            # 下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(img_url) as resp:
                    self.logger.debug(f"图片下载响应状态码: {resp.status}")
                    if resp.status == 200:
                        # 生成唯一文件名
                        filename = f"{emotion_en}_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
                        file_path = os.path.join(emotion_path, filename)
                        
                        # 保存图片
                        content = await resp.read()
                        if not content:
                            return f"下载的图片内容为空"
                            
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        self.logger.info(f"图片已成功保存: {file_path}")
                        return file_path
                    else:
                        error_msg = f"下载图片失败: HTTP {resp.status}"
                        self.logger.error(error_msg)
                        return error_msg
        except aiohttp.ClientError as e:
            error_msg = f"网络请求失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        except IOError as e:
            error_msg = f"文件写入失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"保存图片时发生未知错误: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def call_baidu_ai(self, img_urls):
        """调用百度 AI 进行图片识别"""
        api_key = self.config.get("api_key", "")
        if not api_key:
            raise ValueError("API Key 未配置。请在配置文件中设置。")

        url = "https://qianfan.baidubce.com/v2/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        # 构建消息内容，要求返回简单的情绪描述
        content = [
            {
                "type": "text",
                "text": "请分析这张图片表达的情绪，只需要用一个词回答，例如：开心、悲伤、生气、惊讶等。请确保回答的情绪在以下列表中："
            }
        ]
        
        # 添加图片
        for img_url in img_urls:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            })

        # 从配置文件中获取情绪列表
        emotions_list = list(self.emotion_map.keys())
        content[0]["text"] += "、".join(emotions_list)

        payload = {
            "model": "deepseek-vl2",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    error_data = await resp.text()
                    self.logger.debug(f"响应状态码: {resp.status}")
                    self.logger.debug(f"响应内容: {error_data}")

                    if resp.status != 200:
                        error_msg = f"百度 AI 响应错误: 状态码 {resp.status}"
                        if error_data:
                            error_msg += f", 响应: {error_data}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
                    
                    data = await resp.json()
                    if "error" in data:
                        raise Exception(f"API 调用失败: {data['error'].get('message', '未知错误')}")
                    
                    emotion = data['choices'][0]['message']['content'].strip()
                    result = [f"识别到的情绪是：{emotion}"]
                    
                    # 如果识别出的情绪在表情映射中，保存图片
                    if emotion in self.emotion_map:
                        save_result = await self._save_image(img_urls[0], emotion)
                        if os.path.exists(str(save_result)):
                            result.append(f"✅ 图片已成功保存到：{save_result}")
                        else:
                            result.append(f"❌ 图片保存失败：{save_result}")
                    else:
                        result.append(f"⚠️ 未找到匹配的情绪类型 '{emotion}'，无法保存图片")
                    
                    return "\n".join(result)

        except aiohttp.ClientError as e:
            error_msg = f"网络请求失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
