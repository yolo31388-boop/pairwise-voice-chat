"""
语音通讯系统 - 验收测试

运行方式：python -m pytest tests/test_voice_chat.py -q
共 12 个测试用例
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.voice_chat import VoiceChat, QUICK_CHAT_MESSAGES


def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def tone(amplitude, n=1000):
    """固定幅值的模拟语音帧。"""
    return [amplitude] * n


class TestVoiceChat:
    """语音通讯系统测试套件"""

    def setup_method(self):
        """每个测试前的初始化"""
        self.system = VoiceChat({"mute_store_path": None})

    # ---------- 队伍语音：VAD + AGC ----------

    def test_case_01(self):
        """VAD：背景噪音不传输，正常说话正常传输"""
        noise = [0.005, -0.004, 0.003, -0.005, 0.002] * 200
        assert self.system.team_voice("p1", noise) is None
        speech = tone(0.15)
        frame = self.system.team_voice("p1", speech)
        assert frame is not None
        assert frame["vad_active"] is True
        assert len(frame["samples"]) == len(speech)

    def test_case_02(self):
        """AGC：小声说话被自动放大到目标响度"""
        quiet = tone(0.05)
        frame = self.system.team_voice("p1", quiet)
        assert frame is not None
        assert frame["gain"] > 1.0
        assert rms(frame["samples"]) > rms(quiet)

    def test_case_03(self):
        """AGC：大声说话被自动压小，避免震耳欲聋"""
        loud = tone(0.9)
        frame = self.system.team_voice("p1", loud)
        assert frame is not None
        assert frame["gain"] < 1.0
        assert rms(frame["samples"]) < rms(loud)

    # ---------- 近距离语音：距离衰减 + 方向感 ----------

    def test_case_04(self):
        """距离衰减：1 米与 100 米音量明显不同，越远越小"""
        speech = tone(0.3)
        near = self.system.proximity_voice(
            "a", "b", speech, (1.0, 0.0), (0.0, 0.0))
        far = self.system.proximity_voice(
            "a", "b", speech, (100.0, 0.0), (0.0, 0.0))
        assert near is not None and far is not None
        assert rms(near["left"]) > rms(far["left"])
        assert near["attenuation"] > far["attenuation"]

    def test_case_05(self):
        """距离衰减：超出最大可听距离完全听不到"""
        frame = self.system.proximity_voice(
            "a", "b", tone(0.5), (1000.0, 0.0), (0.0, 0.0))
        assert frame is None

    def test_case_06(self):
        """方向感：声音在右侧时右声道更响，在左侧时左声道更响"""
        speech = tone(0.4, n=200)
        right_side = self.system.proximity_voice(
            "a", "b", speech, (0.0, 5.0), (0.0, 0.0), listener_facing=0.0)
        left_side = self.system.proximity_voice(
            "a", "b", speech, (0.0, -5.0), (0.0, 0.0), listener_facing=0.0)
        assert rms(right_side["right"]) > rms(right_side["left"])
        assert rms(left_side["left"]) > rms(left_side["right"])

    def test_case_07(self):
        """方向感：正前方声源左右声道平衡"""
        speech = tone(0.4, n=200)
        front = self.system.proximity_voice(
            "a", "b", speech, (5.0, 0.0), (0.0, 0.0), listener_facing=0.0)
        assert abs(rms(front["left"]) - rms(front["right"])) < 1e-9

    # ---------- 语音效果：降噪 / 回声消除 / 压缩 ----------

    def test_case_08(self):
        """降噪：背景噪音被压制，语音主体保留"""
        speech = [0.0 if i % 10 < 5 else 0.4 for i in range(1000)]
        clean = self.system.voice_effects(speech)["samples"]
        speech_after = rms([s for s in clean if abs(s) > 0.3])
        noise_after = rms([s for s in clean if abs(s) <= 0.02])
        assert speech_after > 0.3
        assert noise_after < 0.01

    def test_case_09(self):
        """回声消除：自己声音从扬声器泄出的回声被抵消"""
        mic = [0.5] * 200       # 麦克风录到：只有扬声器回声
        speaker = [0.5] * 200   # 扬声器参考信号（自己的声音）
        clean = self.system.echo_cancellation(mic, speaker)
        assert rms(clean) < 0.05

    def test_case_10(self):
        """压缩：μ-law 压缩数据量显著降低且可还原"""
        samples = [(i % 100) / 100.0 - 0.5 for i in range(800)]
        result = self.system.voice_effects(samples)
        compressed = result["compressed"]
        assert len(compressed) == len(samples)
        assert result["compressed_bytes"] < result["original_bytes"]
        decoded = self.system.decompress(compressed)
        assert len(decoded) == len(samples)
        assert rms(decoded) > 0.1

    # ---------- 快捷语音：冷却 / 匹配 / 多语言 ----------

    def test_case_11(self):
        """快捷语音：冷却生效、文本与语音匹配、支持多语言"""
        chat = VoiceChat({
            "mute_store_path": None,
            "quick_chat_cooldown": 2.0,
        })
        first = chat.quick_chat("p1", "need_backup", "zh", now=100.0)
        assert first["ok"] is True
        # 冷却期内刷屏被拒绝
        blocked = chat.quick_chat("p1", "enemy_here", "zh", now=101.0)
        assert blocked["ok"] is False
        assert blocked["reason"] == "cooldown"
        # 冷却过后可发送
        later = chat.quick_chat("p1", "enemy_here", "zh", now=103.0)
        assert later["ok"] is True
        # 文本与语音严格匹配（按需要支援不会播成发现敌人）
        assert "支援" in first["text"]
        assert first["voice"] == "voice_need_backup_zh"
        assert "敌人" in later["text"]
        assert later["voice"] == "voice_enemy_here_zh"
        # 多语言
        en = chat.quick_chat("p2", "need_backup", "en", now=0.0)
        ja = chat.quick_chat("p3", "need_backup", "ja", now=0.0)
        assert en["text"] == QUICK_CHAT_MESSAGES["need_backup"]["en"]
        assert ja["text"] == QUICK_CHAT_MESSAGES["need_backup"]["ja"]
        assert en["voice"].endswith("_en") and ja["voice"].endswith("_ja")

    # ---------- 静音：全频道 / 持久化 / 反馈 ----------

    def test_case_12(self):
        """静音：队伍+近距离双频道生效、持久化、有操作反馈"""
        store = os.path.join(os.path.dirname(__file__), "_tmp_mutes.json")
        if os.path.exists(store):
            os.remove(store)
        try:
            chat = VoiceChat({"mute_store_path": store})
            feedback = chat.voice_muting("listener", "noise_guy", True)
            # 反馈
            assert feedback["ok"] is True
            assert feedback["muted"] is True
            assert feedback["target_id"] == "noise_guy"
            assert feedback["message"]
            assert chat.is_muted("listener", "noise_guy") is True
            # 队伍语音被静音
            assert chat.team_voice(
                "noise_guy", tone(0.3), listener_id="listener") is None
            # 近距离语音也被静音
            assert chat.proximity_voice(
                "noise_guy", "listener", tone(0.3),
                (1.0, 0.0), (0.0, 0.0)) is None
            # 未静音目标不受影响
            assert chat.team_voice(
                "friend", tone(0.3), listener_id="listener") is not None
            # 持久化：重启后静音列表仍在
            chat2 = VoiceChat({"mute_store_path": store})
            assert chat2.is_muted("listener", "noise_guy") is True
            # 取消静音后恢复且带反馈
            off = chat2.voice_muting("listener", "noise_guy", False)
            assert off["muted"] is False
            assert chat2.team_voice(
                "noise_guy", tone(0.3), listener_id="listener") is not None
            chat3 = VoiceChat({"mute_store_path": store})
            assert chat3.is_muted("listener", "noise_guy") is False
        finally:
            if os.path.exists(store):
                os.remove(store)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
