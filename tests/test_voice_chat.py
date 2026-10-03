"""
语音通讯系统 - 验收测试

运行方式：python -m pytest tests/test_voice_chat.py -q
共 12 个测试用例
"""
import math
import os
import random
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.voice_chat import VoiceChat

SAMPLE_RATE = 16000


def sine_wave(freq, amplitude, n):
    return [amplitude * math.sin(2.0 * math.pi * freq * i / SAMPLE_RATE)
            for i in range(n)]


def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class TestVoiceChat:
    """语音通讯系统测试套件"""

    def setup_method(self):
        """每个测试前的初始化"""
        self.system = VoiceChat()

    def test_case_01(self):
        """VAD：无人说话时的背景噪音不应被传输"""
        random.seed(1)
        background_noise = [random.uniform(-0.005, 0.005) for _ in range(1600)]
        assert self.system.team_voice("p1", background_noise) is None

    def test_case_02(self):
        """VAD：真实语音应正常通过并完整返回"""
        speech = sine_wave(440, 0.3, 1600)
        out = self.system.team_voice("p1", speech)
        assert out is not None
        assert len(out) == len(speech)

    def test_case_03(self):
        """AGC：小声玩家被放大、大声玩家被压小，输出电平归一"""
        target = self.system.config["agc_target_rms"]
        quiet_out = self.system.team_voice("quiet", sine_wave(440, 0.05, 1600))
        loud_out = self.system.team_voice("loud", sine_wave(440, 0.9, 1600))
        assert rms(quiet_out) > 0.05  # 小声被放大
        assert rms(loud_out) < 0.9    # 大声被压小
        assert abs(rms(quiet_out) - target) < target * 0.1
        assert abs(rms(loud_out) - target) < target * 0.1

    def test_case_04(self):
        """近距离语音：距离越远音量越小，超出最大距离听不到"""
        speech = sine_wave(440, 0.5, 800)
        self.system.set_position("speaker", 0, 0)
        self.system.set_position("near", 2, 0)
        self.system.set_position("far", 20, 0)
        self.system.set_position("too_far", 100, 0)
        near = self.system.proximity_voice("speaker", "near", speech)
        far = self.system.proximity_voice("speaker", "far", speech)
        assert near["gain"] > far["gain"]
        assert rms(near["left"]) > rms(far["left"])
        assert self.system.proximity_voice("speaker", "too_far", speech) is None

    def test_case_05(self):
        """近距离语音：方向感，声音应偏向说话者所在一侧的声道"""
        speech = sine_wave(440, 0.5, 800)
        self.system.set_position("listener", 0, 0, facing=0.0)
        self.system.set_position("right_side", 0, -10)  # 听者右侧
        self.system.set_position("left_side", 0, 10)    # 听者左侧
        right = self.system.proximity_voice("right_side", "listener", speech)
        left = self.system.proximity_voice("left_side", "listener", speech)
        assert rms(right["right"]) > rms(right["left"])
        assert rms(left["left"]) > rms(left["right"])

    def test_case_06(self):
        """降噪：背景噪音被压制，语音主体保留"""
        random.seed(2)
        # 前段只有背景噪音（未说话），后段是语音+背景噪音
        noise_only = [random.uniform(-0.03, 0.03) for _ in range(800)]
        speech = sine_wave(440, 0.4, 800)
        noisy_speech = [s + random.uniform(-0.03, 0.03) for s in speech]
        cleaned = self.system.reduce_noise(noise_only + noisy_speech)
        assert rms(cleaned[:800]) < rms(noise_only)  # 噪音段被压制
        assert rms(cleaned[800:]) > rms(speech) * 0.5  # 语音段保留

    def test_case_07(self):
        """回声消除：扬声器回传的声音被从麦克风中消除"""
        reference = [a + b for a, b in zip(sine_wave(300, 0.3, 1600),
                                           sine_wave(700, 0.2, 1600))]
        clean_voice = sine_wave(500, 0.3, 1600)
        delay, echo_gain = 20, 0.6
        mic = list(clean_voice)
        for i in range(delay, len(mic)):
            mic[i] += echo_gain * reference[i - delay]
        out = self.system.cancel_echo(mic, reference)
        err_before = rms([m - c for m, c in zip(mic, clean_voice)])
        err_after = rms([o - c for o, c in zip(out, clean_voice)])
        assert err_after < err_before * 0.5

    def test_case_08(self):
        """压缩：语音数据量显著减小且可无损近似还原"""
        samples = sine_wave(440, 0.5, 1600)
        blob = self.system.compress(samples)
        assert len(blob) < len(samples) * 8  # 原始 float64 的 1/8 以下
        restored = self.system.decompress(blob)
        assert len(restored) == len(samples)
        assert max(abs(a - b) for a, b in zip(samples, restored)) < 1e-3

    def test_case_09(self):
        """快捷语音：冷却期内不能刷屏，冷却结束后恢复"""
        system = VoiceChat({"quick_chat_cooldown": 0.1})
        assert system.quick_chat("p1", "need_backup") is not None
        assert system.quick_chat("p1", "need_backup") is None  # 冷却中被拒绝
        time.sleep(0.15)
        assert system.quick_chat("p1", "need_backup") is not None

    def test_case_10(self):
        """快捷语音：文本与语音匹配同一指令，且支持多语言"""
        en = self.system.quick_chat("p1", "need_backup", language="en")
        assert en["command"] == "need_backup"
        assert "need_backup" in en["voice"]
        assert en["text"] == "Need backup!"
        zh = self.system.quick_chat("p2", "need_backup", language="zh")
        assert zh["text"] == "需要支援！"
        assert zh["voice"].endswith("_zh")
        assert "enemy" not in zh["voice"]  # 按"需要支援"不会播"敌人在这里"

    def test_case_11(self):
        """静音：队伍语音与近距离语音同时生效，且有操作反馈"""
        with tempfile.TemporaryDirectory() as tmp:
            system = VoiceChat({"mute_list_path": os.path.join(tmp, "m.json")})
            speech = sine_wave(440, 0.3, 800)
            system.set_position("troll", 1, 0)
            system.set_position("me", 0, 0)
            result = system.mute_player("me", "troll")
            assert result["muted"] is True
            assert result["feedback"]  # 静音时有反馈
            assert system.is_muted("me", "troll")
            assert system.team_voice("troll", speech, listener_id="me") is None
            assert system.proximity_voice("troll", "me", speech) is None
            # 未被静音的玩家不受影响
            assert system.team_voice("friend", speech, listener_id="me") is not None

    def test_case_12(self):
        """静音：静音列表持久化，重启后依然生效，解除静音同样持久"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mutes.json")
            first = VoiceChat({"mute_list_path": path})
            first.mute_player("me", "troll")
            # 模拟重启：新建实例加载同一持久化文件
            second = VoiceChat({"mute_list_path": path})
            assert second.is_muted("me", "troll")
            second.unmute_player("me", "troll")
            third = VoiceChat({"mute_list_path": path})
            assert not third.is_muted("me", "troll")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
