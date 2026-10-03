# 语音通讯系统

## 项目描述

语音通讯系统的bug修复项目。核心模块 `audio/voice_chat.py` 存在多个已知bug，需要修复后使测试全部通过。

## 已知问题

- 队伍语音无VAD无自动增益控制
- 近距离语音完全没做无距离衰减
- 语音无降噪无回声消除无压缩
- 快捷语音无冷却且文本语音不匹配
- 静音功能不全面不持久化无反馈

## 项目结构

```
├── audio/voice_chat.py       # 核心模块（带bug）
├── tests/test_voice_chat.py    # 验收测试（红态）
├── pyproject.toml          # 项目配置
└── README.md               # 说明文档
```

## 运行测试

```bash
python -m pytest tests/test_voice_chat.py -q
```

修复目标：所有 12 个测试全部通过。

## 技术要求

- 纯Python标准库，不引入第三方依赖
- 不修改测试文件
- 保持现有API接口不变
- 修复所有已知bug
