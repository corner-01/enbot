import json
import redis
import logging
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    message_to_dict,
    messages_from_dict,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisChatHistory:
    """
    基于 Redis 的对话历史，Redis 不可用时自动使用进程内存。

    短期历史保留 24 小时，可选的长期摘要保留 7 天。
    """
    def __init__(self, session_id: str, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0, **kwargs):
        """
        初始化 Redis 连接和会话 ID。

        :param session_id: 会话唯一标识符，用于区分不同用户的对话
        :param redis_host: Redis 主机地址
        :param redis_port: Redis 端口
        :param redis_db: Redis 数据库编号
        :param kwargs: 其他传递给 redis.StrictRedis 的参数
        """
        self.session_id = session_id
        self.redis_key = f"chat_history:{session_id}"
        self.long_term_key = f"long_term_memory:{session_id}"
        self._local_memory_store = []

        try:
            # 添加连接池配置
            pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                **kwargs
            )
            self.redis_client = redis.StrictRedis(connection_pool=pool)
            self.redis_client.ping()
            logger.info(f"成功连接到 Redis (Session: {session_id})")
        except redis.ConnectionError as e:
            logger.error(f"无法连接到 Redis 服务器: {e}")
            # 如果连接失败，使用本地内存作为降级方案
            self.redis_client = None
            self._local_memory_store = []
            logger.warning("Redis 不可用，切换至本地内存模式。")

    def _message_to_dict(self, message: BaseMessage) -> dict:
        """将 LangChain 消息对象转换为字典以便存储"""
        return message_to_dict(message)

    def _dict_to_message(self, message_dict: dict) -> BaseMessage:
        """将字典转换回 LangChain 消息对象"""
        # 兼容修复前已经写入 Redis 的扁平消息结构。
        if "data" not in message_dict:
            message_type = message_dict.get("type", "")
            if message_type == "summary":
                return AIMessage(content=message_dict.get("content", ""))
            message_dict = {
                "type": message_type,
                "data": {
                    key: value
                    for key, value in message_dict.items()
                    if key != "type"
                },
            }
        return messages_from_dict([message_dict])[0]

    def add_message(self, message: BaseMessage) -> bool:
        """
        添加一条消息到历史记录。

        :param message: BaseMessage 对象 (HumanMessage 或 AIMessage)
        :return: 是否成功
        """
        if self.redis_client:
            try:
                # 使用 List 结构存储，RPUSH 保证最新的在尾部
                self.redis_client.rpush(self.redis_key, json.dumps(self._message_to_dict(message)))
                # 设置过期时间，例如 24 小时 (86400 秒)
                self.redis_client.expire(self.redis_key, 86400)
                return True
            except redis.RedisError as e:
                logger.error(f"Redis 写入错误: {e}")
                # 降级到本地存储
                self._local_memory_store = self.get_messages()  # 先获取现有消息
                self._local_memory_store.append(message)  # 添加新消息
                return False
        else:
            # 本地降级处理
            self._local_memory_store.append(message)
            return True

    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        获取当前会话的历史消息。

        :param limit: 返回的消息数量限制，None 表示返回全部
        :return: BaseMessage 对象列表
        """
        if self.redis_client:
            try:
                # 获取列表长度
                length = self.redis_client.llen(self.redis_key)
                if length == 0:
                    return []

                # 确定获取范围
                start = 0 if limit is None else max(0, length - limit)

                # 获取数据
                raw_messages = self.redis_client.lrange(self.redis_key, start, -1)

                # 反序列化
                messages = []
                for raw_msg in raw_messages:
                    try:
                        msg_dict = json.loads(raw_msg)
                        messages.append(self._dict_to_message(msg_dict))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        logger.warning(f"跳过无法解析的 Redis 消息: {e}")
                        continue
                return messages
            except redis.RedisError as e:
                logger.error(f"Redis 读取错误: {e}")
                # 降级到本地存储
                if limit:
                    return self._local_memory_store[-limit:] if self._local_memory_store else []
                return self._local_memory_store if self._local_memory_store else []
        else:
            # 本地降级处理
            if limit:
                return self._local_memory_store[-limit:] if self._local_memory_store else []
            return self._local_memory_store if self._local_memory_store else []

    @property
    def messages(self) -> List[BaseMessage]:
        """
        提供与 get_messages() 相同的功能，但作为属性访问
        """
        return self.get_messages()

    def get_long_term_memory(self, limit: int = 5) -> List[BaseMessage]:
        """
        获取长期记忆（例如之前的对话摘要或重要信息）。
        这里简单实现为获取 Redis 中另一个 key 的数据，实际业务可能更复杂。

        :param limit: 返回的数量
        :return: BaseMessage 对象列表
        """
        if self.redis_client:
            try:
                # 尝试获取长期记忆 key
                length = self.redis_client.llen(self.long_term_key)
                if length == 0:
                    return []

                start = 0 if limit is None else max(0, length - limit)
                raw_messages = self.redis_client.lrange(self.long_term_key, start, -1)

                messages = []
                for raw_msg in raw_messages:
                    try:
                        msg_dict = json.loads(raw_msg)
                        messages.append(self._dict_to_message(msg_dict))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        logger.warning(f"跳过无法解析的长期记忆: {e}")
                        continue
                return messages
            except redis.RedisError as e:
                logger.error(f"读取长期记忆错误: {e}")
                return []
        return []

    def sync_long_term_memory(self) -> bool:
        """
        将短期记忆中的重要信息同步到长期记忆。
        这是一个简单的实现，实际业务可能需要更复杂的逻辑。

        :return: 是否成功同步
        """
        if not self.redis_client:
            return False

        try:
            # 获取最近的对话
            recent_messages = self.get_messages(limit=10)
            if not recent_messages:
                return False

            # 这里可以添加更复杂的逻辑来判断哪些信息需要长期保存
            # 例如：提取重要信息、关键决策等
            # 这里简单实现为保存最近的对话摘要
            summary = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])

            # 添加到长期记忆
            self.redis_client.rpush(self.long_term_key, json.dumps({
                "type": "summary",
                "content": summary,
                "timestamp": datetime.now().isoformat()
            }))

            # 设置长期记忆的过期时间（例如 7 天）
            self.redis_client.expire(self.long_term_key, 604800)
            return True

        except Exception as e:
            logger.error(f"同步长期记忆失败: {e}")
            return False

    def clear_session(self) -> bool:
        """清空当前会话的短期记忆"""
        if self.redis_client:
            try:
                self.redis_client.delete(self.redis_key)
                return True
            except redis.RedisError as e:
                logger.error(f"清空会话错误: {e}")
                return False
        else:
            self._local_memory_store = []
            return True
