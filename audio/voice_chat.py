"""
语音通讯系统 - 核心模块

功能：
- 队伍语音：VAD 语音活动检测 + 自动增益控制（AGC）
- 近距离语音：距离衰减 + 立体声方向感
- 语音效果：降噪 + 回声消除 + 压缩
- 快捷语音：冷却 + 文本语音匹配 + 多语言
- 语音静音：队伍/近距离全频道生效 + 持久化 + 反馈
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import array
import json
import math
import os
import time
import zlib


DEFAULT_CONFIG = {
    "sample_rate": 16000,
    # VAD：低于该 RMS 能量的帧视为非语音，不传输
    "vad_threshold": 0.02,
    # AGC：目标 RMS 与增益范围
    "agc_target_rms": 0.3,
    "agc_min_gain": 0.1,
    "agc_max_gain": 10.0,
    # 近距离语音：参考距离（该距离内音量不衰减）与最大可听距离
    "reference_distance": 1.0,
    "max_voice_distance": 50.0,
    # 降噪：噪声门限与衰减强度
    "noise_reduction": 0.75,
    # 回声消除：最大搜索延迟（采样点数）
    "echo_max_delay": 64,
    # 快捷语音：冷却时间（秒）与默认语言
    "quick_chat_cooldown": 3.0,
    "language": "en",
    # 静音列表持久化路径
    "mute_list_path": "mute_list.json",
}


# 快捷语音指令表：文本与语音按同一指令键生成，保证永远匹配
QUICK_CHAT_COMMANDS = {
    "need_backup": {
        "voice": "voice_need_backup",
        "text": {
            "en": "Need backup!",
            "zh": "需要支援！",
            "ja": "支援が必要だ！",
        },
    },
    "enemy_here": {
        "voice": "voice_enemy_here",
        "text": {
            "en": "Enemy spotted!",
            "zh": "敌人在这里！",
            "ja": "敵を発見！",
        },
    },
    "retreat": {
        "voice": "voice_retreat",
        "text": {
            "en": "Fall back!",
            "zh": "撤退！",
            "ja": "退却しろ！",
        },
    },
    "nice_shot": {
        "voice": "voice_nice_shot",
        "text": {
            "en": "Nice shot!",
            "zh": "好枪法！",
            "ja": "ナイスショット！",
        },
    },
}

SUPPORTED_LANGUAGES = ("en", "zh", "ja")


@dataclass
class Config:
    """配置项"""
    pass


def _rms(samples: List[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class VoiceChat:
    """语音通讯系统核心类"""

    def __init__(self, config: Optional[Dict] = None):
        merged = dict(DEFAULT_CONFIG)
        if config:
            merged.update(config)
        self.config = merged
        self._state = {}
        self._history = []
        self._positions: Dict[str, Tuple[float, float, float, float]] = {}
        self._quick_chat_last: Dict[str, float] = {}
        self._mutes: Dict[str, set] = {}
        self._load_mutes()

    def update(self, dt: float):
        """每帧更新"""
        pass

    def reset(self):
        """重置状态"""
        self._state = {}
        self._history = []
        self._positions = {}
        self._quick_chat_last = {}

    # ------------------------------------------------------------------
    # 队伍语音：VAD + AGC
    # ------------------------------------------------------------------

    def vad_detect(self, samples: List[float]) -> bool:
        """语音活动检测：帧能量超过阈值才视为语音"""
        return _rms(samples) >= self.config["vad_threshold"]

    def apply_agc(self, samples: List[float]) -> List[float]:
        """自动增益控制：把语音 RMS 归一到目标电平"""
        rms = _rms(samples)
        if rms <= 1e-9:
            return list(samples)
        gain = _clamp(self.config["agc_target_rms"] / rms,
                      self.config["agc_min_gain"],
                      self.config["agc_max_gain"])
        return [_clamp(s * gain, -1.0, 1.0) for s in samples]

    def team_voice(self, speaker_id: str, samples: List[float],
                   listener_id: Optional[str] = None) -> Optional[List[float]]:
        """队伍语音处理：先检查静音，再 VAD 门控，最后 AGC"""
        if listener_id is not None and self.is_muted(listener_id, speaker_id):
            return None
        if not self.vad_detect(samples):
            return None
        return self.apply_agc(samples)

    # ------------------------------------------------------------------
    # 近距离语音：距离衰减 + 方向感
    # ------------------------------------------------------------------

    def set_position(self, player_id: str, x: float, y: float,
                     z: float = 0.0, facing: float = 0.0):
        """设置玩家位置与朝向（facing 为 xy 平面弧度，0 表示 +x 方向）"""
        self._positions[player_id] = (x, y, z, facing)

    def compute_attenuation(self, distance: float) -> float:
        """距离衰减：参考距离内不衰减，之后按 1/d 衰减，超出最大距离为 0"""
        ref = self.config["reference_distance"]
        if distance > self.config["max_voice_distance"]:
            return 0.0
        return min(1.0, ref / max(distance, ref))

    def compute_pan(self, relative_angle: float) -> Tuple[float, float]:
        """方向感：根据说话者相对听者朝向的方位角计算左右声道增益（等功率声像）"""
        pan = -math.sin(relative_angle)  # +1 正右方，-1 正左方
        theta = (pan + 1.0) * math.pi / 4.0
        return math.cos(theta), math.sin(theta)

    def proximity_voice(self, speaker_id: str, listener_id: str,
                        samples: List[float]) -> Optional[Dict]:
        """近距离语音：按距离衰减并按方位分配到左右声道"""
        if self.is_muted(listener_id, speaker_id):
            return None
        if speaker_id not in self._positions or listener_id not in self._positions:
            raise KeyError("说话者和听者都需要先通过 set_position 注册位置")
        sx, sy, sz, _ = self._positions[speaker_id]
        lx, ly, lz, facing = self._positions[listener_id]
        distance = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2 + (sz - lz) ** 2)
        gain = self.compute_attenuation(distance)
        if gain <= 0.0:
            return None
        angle = math.atan2(sy - ly, sx - lx)
        left_pan, right_pan = self.compute_pan(angle - facing)
        left_gain = gain * left_pan
        right_gain = gain * right_pan
        return {
            "left": [s * left_gain for s in samples],
            "right": [s * right_gain for s in samples],
            "gain": gain,
            "distance": distance,
        }

    # ------------------------------------------------------------------
    # 语音效果：降噪 + 回声消除 + 压缩
    # ------------------------------------------------------------------

    def reduce_noise(self, samples: List[float]) -> List[float]:
        """降噪：估计噪声底并门控衰减，再做平滑滤波去除高频噪声"""
        if not samples:
            return []
        strength = self.config["noise_reduction"]
        frame = max(1, len(samples) // 16)
        frames = [samples[i:i + frame]
                  for i in range(0, len(samples), frame)]
        noise_floor = min(_rms(f) for f in frames)
        # 按帧能量门控：整帧能量接近噪声底才衰减，避免损伤语音帧
        gated = []
        for f in frames:
            if _rms(f) <= noise_floor * 1.5:
                gated.extend(s * (1.0 - strength) for s in f)
            else:
                gated.extend(f)
        smoothed = []
        for i in range(len(gated)):
            window = gated[max(0, i - 1):i + 2]
            smoothed.append(sum(window) / len(window))
        return smoothed

    def cancel_echo(self, mic: List[float],
                    reference: List[float]) -> List[float]:
        """回声消除：搜索参考信号的最佳延迟与增益并从麦克风中减去"""
        if not mic or not reference:
            return list(mic)
        max_delay = min(self.config["echo_max_delay"], len(reference) - 1)
        best_gain, best_delay, best_residual = 0.0, 0, float("inf")
        for delay in range(max_delay + 1):
            num = den = 0.0
            for i in range(delay, min(len(mic), len(reference) + delay)):
                ref = reference[i - delay]
                num += mic[i] * ref
                den += ref * ref
            if den <= 1e-12:
                continue
            gain = num / den
            residual = 0.0
            for i in range(delay, min(len(mic), len(reference) + delay)):
                diff = mic[i] - gain * reference[i - delay]
                residual += diff * diff
            if residual < best_residual:
                best_residual, best_gain, best_delay = residual, gain, delay
        out = list(mic)
        for i in range(best_delay, min(len(out), len(reference) + best_delay)):
            out[i] -= best_gain * reference[i - best_delay]
        return out

    def voice_effects(self, samples: List[float],
                      reference: Optional[List[float]] = None) -> List[float]:
        """完整语音效果链：回声消除（如有扬声器参考信号）+ 降噪"""
        out = list(samples)
        if reference:
            out = self.cancel_echo(out, reference)
        return self.reduce_noise(out)

    def compress(self, samples: List[float]) -> bytes:
        """压缩：16bit 量化 + zlib，减小网络传输数据量"""
        pcm = array.array("h", (int(_clamp(s, -1.0, 1.0) * 32767)
                                for s in samples))
        return zlib.compress(pcm.tobytes())

    def decompress(self, blob: bytes) -> List[float]:
        """解压缩：compress 的逆过程"""
        pcm = array.array("h")
        pcm.frombytes(zlib.decompress(blob))
        return [v / 32767.0 for v in pcm]

    # ------------------------------------------------------------------
    # 快捷语音：冷却 + 文本语音匹配 + 多语言
    # ------------------------------------------------------------------

    def quick_chat(self, player_id: str, command: str,
                   language: Optional[str] = None,
                   now: Optional[float] = None) -> Optional[Dict]:
        """发送快捷语音；冷却期内返回 None；文本与语音按指令键匹配"""
        if command not in QUICK_CHAT_COMMANDS:
            raise ValueError(f"未知快捷语音指令: {command}")
        lang = language or self.config["language"]
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"
        now = time.monotonic() if now is None else now
        last = self._quick_chat_last.get(player_id, float("-inf"))
        if now - last < self.config["quick_chat_cooldown"]:
            return None
        self._quick_chat_last[player_id] = now
        cmd = QUICK_CHAT_COMMANDS[command]
        return {
            "command": command,
            "text": cmd["text"][lang],
            "voice": f"{cmd['voice']}_{lang}",
            "language": lang,
        }

    # ------------------------------------------------------------------
    # 语音静音：全频道生效 + 持久化 + 反馈
    # ------------------------------------------------------------------

    def _mute_path(self) -> str:
        return self.config["mute_list_path"]

    def _load_mutes(self):
        path = self._mute_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._mutes = {k: set(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                self._mutes = {}

    def _save_mutes(self):
        path = self._mute_path()
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({k: sorted(v) for k, v in self._mutes.items()}, fh)

    def mute_player(self, muter_id: str, target_id: str) -> Dict:
        """静音目标玩家（队伍语音与近距离语音同时生效），返回操作反馈"""
        self._mutes.setdefault(muter_id, set()).add(target_id)
        self._save_mutes()
        feedback = {"muted": True, "target": target_id,
                    "feedback": f"已静音玩家 {target_id}（队伍语音与近距离语音）"}
        self._history.append(feedback)
        return feedback

    def unmute_player(self, muter_id: str, target_id: str) -> Dict:
        """解除静音，返回操作反馈"""
        self._mutes.get(muter_id, set()).discard(target_id)
        self._save_mutes()
        feedback = {"muted": False, "target": target_id,
                    "feedback": f"已解除对玩家 {target_id} 的静音"}
        self._history.append(feedback)
        return feedback

    def is_muted(self, muter_id: str, target_id: str) -> bool:
        """查询目标是否被静音（队伍语音与近距离语音共用）"""
        return target_id in self._mutes.get(muter_id, set())
