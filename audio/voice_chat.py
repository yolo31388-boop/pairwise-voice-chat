"""
语音通讯系统 - 核心模块（修复版）

功能：
- 队伍语音：VAD 语音活动检测 + AGC 自动增益控制
- 近距离语音：距离衰减 + 立体声方向感（声像定位）
- 语音效果：降噪 + 回声消除 + μ-law 压缩
- 快捷语音：冷却时间 + 文本/语音严格匹配 + 多语言
- 语音静音：全频道生效 + 持久化 + 操作反馈
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math
import os
import time


# ---------------------------------------------------------------- 常量

# 快捷语音消息表：message_id -> {language: text}
# 语音资源 id 由 message_id + language 派生，保证文本与语音永远匹配
QUICK_CHAT_MESSAGES: Dict[str, Dict[str, str]] = {
    "need_backup": {
        "en": "Need backup!",
        "zh": "需要支援！",
        "ja": "支援が必要だ！",
    },
    "enemy_here": {
        "en": "Enemy spotted!",
        "zh": "发现敌人！",
        "ja": "敵を発見！",
    },
    "retreat": {
        "en": "Fall back!",
        "zh": "撤退！",
        "ja": "撤退しろ！",
    },
    "nice_shot": {
        "en": "Nice shot!",
        "zh": "好枪法！",
        "ja": "ナイスショット！",
    },
}

SUPPORTED_LANGUAGES = ("en", "zh", "ja")
DEFAULT_LANGUAGE = "en"

_MULAW_MAX = 0x7FFF
_MULAW_BIAS = 0x84  # 132


@dataclass
class Config:
    """配置项"""
    vad_threshold: float = 0.02        # VAD 能量阈值（RMS）
    agc_target_rms: float = 0.2        # AGC 目标响度
    agc_min_gain: float = 0.1          # AGC 最小增益
    agc_max_gain: float = 10.0         # AGC 最大增益
    proximity_max_range: float = 150.0  # 近距离语音最大可听距离（米）
    proximity_ref_distance: float = 2.0  # 距离衰减参考距离（米）
    noise_gate_ratio: float = 0.1      # 降噪门限衰减比
    echo_gain: float = 0.9             # 回声消除参考信号增益
    quick_chat_cooldown: float = 2.0   # 快捷语音冷却（秒）
    mute_store_path: Optional[str] = "voice_mutes.json"  # 静音列表持久化路径


class VoiceChat:
    """语音通讯系统核心类"""

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        self.config = cfg
        self._cfg = Config(**{k: v for k, v in cfg.items() if k in Config.__dataclass_fields__})
        self._state = {}
        self._history = []
        # 快捷语音冷却：player_id -> 上次发送时间戳
        self._quick_chat_last: Dict[str, float] = {}
        # 静音表：listener_id -> {speaker_id, ...}
        self._mutes: Dict[str, set] = {}
        self._load_mutes()

    # ------------------------------------------------------------ 基础

    def update(self, dt: float):
        """每帧更新"""
        pass

    def reset(self):
        """重置状态（静音列表持久化保留）"""
        self._state = {}
        self._history = []
        self._quick_chat_last = {}

    @staticmethod
    def _rms(samples: List[float]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    @staticmethod
    def _clamp(sample: float) -> float:
        return max(-1.0, min(1.0, sample))

    # ------------------------------------------------------------ 队伍语音

    def team_voice(self, player_id: str, samples: List[float],
                   listener_id: Optional[str] = None,
                   sample_rate: int = 16000) -> Optional[Dict]:
        """队伍语音处理：VAD + AGC。

        - VAD：帧能量低于阈值判定为非语音（背景噪音），不传输，返回 None。
        - AGC：自动增益控制，把响度拉向目标 RMS，小声放大、大声压小。
        - 静音：listener 静音了 player 时不传输，返回 None。
        """
        if listener_id is not None and self.is_muted(listener_id, player_id):
            return None
        if not samples:
            return None
        rms = self._rms(samples)
        if rms < self._cfg.vad_threshold:
            return None  # VAD：无语音活动，不传输
        gain = self._agc_gain(rms)
        processed = [self._clamp(s * gain) for s in samples]
        return {
            "player_id": player_id,
            "samples": processed,
            "sample_rate": sample_rate,
            "gain": gain,
            "vad_active": True,
            "channel": "team",
        }

    def _agc_gain(self, rms: float) -> float:
        """自动增益：目标 RMS / 当前 RMS，限制在 [min, max] 区间。"""
        if rms <= 0.0:
            return 1.0
        gain = self._cfg.agc_target_rms / rms
        return max(self._cfg.agc_min_gain, min(self._cfg.agc_max_gain, gain))

    # ------------------------------------------------------------ 近距离语音

    def proximity_voice(self, speaker_id: str, listener_id: str,
                        samples: List[float],
                        speaker_pos: Tuple[float, float],
                        listener_pos: Tuple[float, float],
                        listener_facing: float = 0.0,
                        sample_rate: int = 16000) -> Optional[Dict]:
        """近距离语音：距离衰减 + 立体声方向感。

        - 超出最大可听距离返回 None（完全听不到）。
        - 距离衰减：音量随距离按 1/(1+d/ref) 衰减。
        - 方向感：按说话者相对听者朝向的方位角做等功率声像（panning），
          左右声道音量不同，玩家可判断声音来向。
        - 静音：listener 静音了 speaker 时返回 None。
        """
        if self.is_muted(listener_id, speaker_id):
            return None
        if not samples:
            return None
        dx = speaker_pos[0] - listener_pos[0]
        dy = speaker_pos[1] - listener_pos[1]
        distance = math.hypot(dx, dy)
        if distance > self._cfg.proximity_max_range:
            return None  # 超出可听范围

        attenuation = 1.0 / (1.0 + distance / self._cfg.proximity_ref_distance)

        # 方位角：说话者相对听者朝向的角度；pan>0 偏右，pan<0 偏左
        if distance > 1e-9:
            angle = math.atan2(dy, dx) - listener_facing
            pan = math.sin(angle)
        else:
            pan = 0.0
        pan = max(-1.0, min(1.0, pan))

        # 等功率声像（constant-power panning）
        theta = (pan + 1.0) * math.pi / 4.0
        left_gain = math.cos(theta)
        right_gain = math.sin(theta)

        left = [self._clamp(s * attenuation * left_gain) for s in samples]
        right = [self._clamp(s * attenuation * right_gain) for s in samples]
        return {
            "speaker_id": speaker_id,
            "listener_id": listener_id,
            "left": left,
            "right": right,
            "distance": distance,
            "attenuation": attenuation,
            "pan": pan,
            "sample_rate": sample_rate,
            "channel": "proximity",
        }

    # ------------------------------------------------------------ 语音效果

    def voice_effects(self, samples: List[float],
                      echo_reference: Optional[List[float]] = None,
                      noise_floor: Optional[float] = None) -> Dict:
        """语音效果链：降噪 -> 回声消除 -> 压缩。

        - 降噪：噪声门，低于噪声底噪的采样大幅衰减，保留语音。
        - 回声消除：用扬声器参考信号抵消麦克风录到的回声。
        - 压缩：μ-law 编码，每个采样压缩为 1 字节，降低网络带宽。
        """
        denoised = self.noise_reduction(samples, noise_floor)
        if echo_reference is not None:
            clean = self.echo_cancellation(denoised, echo_reference)
        else:
            clean = denoised
        compressed = self.compress(clean)
        return {
            "samples": clean,
            "compressed": compressed,
            "noise_floor": noise_floor if noise_floor is not None
                           else self._estimate_noise_floor(samples),
            "echo_cancelled": echo_reference is not None,
            "original_bytes": len(samples) * 8,
            "compressed_bytes": len(compressed),
        }

    def noise_reduction(self, samples: List[float],
                        noise_floor: Optional[float] = None) -> List[float]:
        """降噪：噪声门。低于底噪的采样按比率衰减，背景噪音被压制。"""
        if not samples:
            return []
        floor = noise_floor if noise_floor is not None else self._estimate_noise_floor(samples)
        ratio = self._cfg.noise_gate_ratio
        return [s if abs(s) >= floor else s * ratio for s in samples]

    @staticmethod
    def _estimate_noise_floor(samples: List[float]) -> float:
        """估计噪声底噪：取采样幅值的低分位（约 20% 分位）。"""
        if not samples:
            return 0.0
        mags = sorted(abs(s) for s in samples)
        idx = min(len(mags) - 1, int(len(mags) * 0.2))
        return mags[idx]

    def echo_cancellation(self, mic_samples: List[float],
                          reference: List[float]) -> List[float]:
        """回声消除：从麦克信号中减去扬声器参考信号（回声来源）。"""
        gain = self._cfg.echo_gain
        n = len(mic_samples)
        out = []
        for i in range(n):
            ref = reference[i] if i < len(reference) else 0.0
            out.append(self._clamp(mic_samples[i] - gain * ref))
        return out

    # ------------------------------------------------------------ 压缩（μ-law）

    @staticmethod
    def compress(samples: List[float]) -> bytes:
        """μ-law 压缩：float 采样 -> 1 字节，数据量降为 1/8。"""
        out = bytearray()
        for s in samples:
            out.append(VoiceChat._mulaw_encode(s))
        return bytes(out)

    @staticmethod
    def decompress(data: bytes) -> List[float]:
        """μ-law 解压：1 字节 -> float 采样。"""
        return [VoiceChat._mulaw_decode(b) for b in data]

    @staticmethod
    def _mulaw_encode(sample: float) -> int:
        x = int(max(-1.0, min(1.0, sample)) * _MULAW_MAX)
        sign = 0x80 if x < 0 else 0x00
        if x < 0:
            x = -x
        x = min(x, _MULAW_MAX) + _MULAW_BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (x & mask):
            mask >>= 1
            exponent -= 1
        mantissa = (x >> (exponent + 3)) & 0x0F
        return (~(sign | (exponent << 4) | mantissa)) & 0xFF

    @staticmethod
    def _mulaw_decode(code: int) -> float:
        u = (~code) & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        x = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
        if sign:
            x = -x
        return max(-1.0, min(1.0, x / _MULAW_MAX))

    # ------------------------------------------------------------ 快捷语音

    def quick_chat(self, player_id: str, message_id: str,
                   language: str = DEFAULT_LANGUAGE,
                   now: Optional[float] = None) -> Dict:
        """快捷语音：冷却限制 + 文本/语音匹配 + 多语言。

        - 冷却：同一玩家在冷却时间内重复发送会被拒绝。
        - 匹配：语音资源 id 由 message_id 派生，文本与语音严格对应。
        - 多语言：支持 en/zh/ja，未知语言回退英文。
        """
        if message_id not in QUICK_CHAT_MESSAGES:
            return {"ok": False, "reason": "unknown_message",
                    "message_id": message_id}
        if language not in SUPPORTED_LANGUAGES:
            language = DEFAULT_LANGUAGE

        now = time.time() if now is None else now
        last = self._quick_chat_last.get(player_id)
        cooldown = self._cfg.quick_chat_cooldown
        if last is not None and now - last < cooldown:
            return {"ok": False, "reason": "cooldown",
                    "remaining": cooldown - (now - last)}

        self._quick_chat_last[player_id] = now
        text = QUICK_CHAT_MESSAGES[message_id][language]
        return {
            "ok": True,
            "player_id": player_id,
            "message_id": message_id,
            "text": text,
            "voice": f"voice_{message_id}_{language}",
            "language": language,
        }

    # ------------------------------------------------------------ 静音

    def voice_muting(self, player_id: str, target_id: str,
                     muted: bool = True) -> Dict:
        """静音/取消静音：全频道生效 + 持久化 + 反馈。

        - 全面：对队伍语音和近距离语音同时生效（见 is_muted）。
        - 持久化：静音列表写入 JSON 文件，重启后自动恢复。
        - 反馈：返回操作结果与提示消息。
        """
        targets = self._mutes.setdefault(player_id, set())
        if muted:
            targets.add(target_id)
            message = f"已静音玩家 {target_id}（队伍语音与近距离语音均生效）"
        else:
            targets.discard(target_id)
            message = f"已取消静音玩家 {target_id}"
        self._save_mutes()
        return {
            "ok": True,
            "player_id": player_id,
            "target_id": target_id,
            "muted": muted,
            "message": message,
        }

    def is_muted(self, listener_id: str, speaker_id: str) -> bool:
        """listener 是否静音了 speaker（队伍语音与近距离语音共用）。"""
        return speaker_id in self._mutes.get(listener_id, set())

    def _mute_path(self) -> Optional[str]:
        return self._cfg.mute_store_path

    def _save_mutes(self):
        path = self._mute_path()
        if not path:
            return
        data = {k: sorted(v) for k, v in self._mutes.items() if v}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_mutes(self):
        path = self._mute_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._mutes = {k: set(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError, TypeError):
            self._mutes = {}
