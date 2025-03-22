from io import BytesIO
import os
import pickle
import sys
import traceback
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer
from time import time as ttime
import fairseq
import faiss
import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe

## patched for Plugin !!!
from pathlib import Path
rvc_sts_plugin_dir = Path(Path.cwd() / "Plugins" / "rvc_sts_plugin")

sys.path.append(str(rvc_sts_plugin_dir.resolve()))
sys.path.append(os.path.join(rvc_sts_plugin_dir, "Retrieval-based-Voice-Conversion-WebUI"))

hubert_model_path = os.path.join(rvc_sts_plugin_dir, "Retrieval-based-Voice-Conversion-WebUI", "assets", "hubert", "hubert_base.pt")
rmvpe_model_path = os.path.join(rvc_sts_plugin_dir, "Retrieval-based-Voice-Conversion-WebUI", "assets", "rmvpe", "rmvpe.pt")

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

now_dir = os.getcwd()
sys.path.append(now_dir)
from multiprocessing import Manager as M

from configs.config import Config

# config = Config()

mm = M()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config: Config,
        last_rvc=None,
    ) -> None:
        """
        初始化
        """
        try:
            if config.dml == True:

                def forward_dml(ctx, x, scale):
                    ctx.scale = scale
                    res = x.clone().detach()
                    return res

                fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
            # global config
            self.config = config
            self.inp_q = inp_q
            self.opt_q = opt_q
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.time_step = 160 / 16000 * 1000
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.sr = 16000
            self.window = 160
            self.n_cpu = n_cpu
            self.use_jit = self.config.use_jit
            self.is_half = config.is_half
            self.debug = False

            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
            self.pth_path: str = pth_path
            self.index_path = index_path
            self.index_rate = index_rate

            if last_rvc is None:
                torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
                models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    [hubert_model_path],
                    suffix="",
                )
                hubert_model = models[0]
                hubert_model = hubert_model.to(self.device)
                if self.is_half:
                    hubert_model = hubert_model.half()
                else:
                    hubert_model = hubert_model.float()
                hubert_model.eval()
                self.model = hubert_model
            else:
                self.model = last_rvc.model

            self.net_g: nn.Module = None

            def set_default_model():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            def set_jit_model():
                jit_pth_path = self.pth_path.rstrip(".pth")
                jit_pth_path += ".half.jit" if self.is_half else ".jit"
                reload = False
                if str(self.device) == "cuda":
                    self.device = torch.device("cuda:0")
                if os.path.exists(jit_pth_path):
                    cpt = jit.load(jit_pth_path)
                    model_device = cpt["device"]
                    if model_device != str(self.device):
                        reload = True
                else:
                    reload = True

                if reload:
                    cpt = jit.synthesizer_jit_export(
                        self.pth_path,
                        "script",
                        None,
                        device=self.device,
                        is_half=self.is_half,
                    )

                self.tgt_sr = cpt["config"][-1]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                self.net_g = torch.jit.load(
                    BytesIO(cpt["model"]), map_location=self.device
                )
                self.net_g.infer = self.net_g.forward
                self.net_g.eval().to(self.device)

            def set_synthesizer():
                if self.use_jit and not config.dml:
                    if self.is_half and "cpu" in str(self.device):
                        printt(
                            "Use default Synthesizer model. \
                                    Jit is not supported on the CPU for half floating point"
                        )
                        set_default_model()
                    else:
                        set_jit_model()
                else:
                    set_default_model()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                if last_rvc.use_jit != self.use_jit:
                    set_synthesizer()
                else:
                    self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("Index search enabled")
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        f0_min = self.f0_min
        f0_max = self.f0_max
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def get_f0(self, x, f0_up_key, n_cpu, method="harvest"):
        n_cpu = int(n_cpu)
        if method == "crepe":
            return self.get_f0_crepe(x, f0_up_key)
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "pm":
            p_len = x.shape[0] // 160 + 1
            f0 = (
                parselmouth.Sound(x, 16000)
                .to_pitch_ac(
                    time_step=0.01,
                    voicing_threshold=0.6,
                    pitch_floor=50,
                    pitch_ceiling=1100,
                )
                .selected_array["frequency"]
            )

            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                # printt(pad_size, p_len - len(f0) - pad_size)
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )

            f0 *= pow(2, f0_up_key / 12)
            return self.get_f0_post(f0)
        if n_cpu == 1:
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            f0 = signal.medfilt(f0, 3)
            f0 *= pow(2, f0_up_key / 12)
            return self.get_f0_post(f0)
        f0bak = np.zeros(x.shape[0] // 160 + 1, dtype=np.float64)
        length = len(x)
        part_length = 160 * ((length // 160 - 1) // n_cpu + 1)
        n_cpu = (length // 160 - 1) // (part_length // 160) + 1
        ts = ttime()
        res_f0 = mm.dict()
        for idx in range(n_cpu):
            tail = part_length * (idx + 1) + 320
            if idx == 0:
                self.inp_q.put((idx, x[:tail], res_f0, n_cpu, ts))
            else:
                self.inp_q.put(
                    (idx, x[part_length * idx - 320 : tail], res_f0, n_cpu, ts)
                )
        while 1:
            res_ts = self.opt_q.get()
            if res_ts == ts:
                break
        f0s = [i[1] for i in sorted(res_f0.items(), key=lambda x: x[0])]
        for idx, f0 in enumerate(f0s):
            if idx == 0:
                f0 = f0[:-3]
            elif idx != n_cpu - 1:
                f0 = f0[2:-3]
            else:
                f0 = f0[2:]
            f0bak[
                part_length * idx // 160 : part_length * idx // 160 + f0.shape[0]
            ] = f0
        f0bak = signal.medfilt(f0bak, 3)
        f0bak *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0bak)

    def get_f0_crepe(self, x, f0_up_key):
        if "privateuseone" in str(self.device):  ###不支持dml，cpu又太慢用不成，拿pm顶替
            return self.get_f0(x, f0_up_key, 1, "pm")
        audio = torch.tensor(np.copy(x))[None].float()
        # printt("using crepe,device:%s"%self.device)
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            160,
            self.f0_min,
            self.f0_max,
            "full",
            batch_size=512,
            # device=self.device if self.device.type!="privateuseone" else "cpu",###crepe不用半精度全部是全精度所以不愁###cpu延迟高到没法用
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def set_debug(self, val: bool):
        self.debug = val

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.lib.rmvpe import RMVPE

            printt("Loading rmvpe model")
            self.model_rmvpe = RMVPE(
                # "rmvpe.pt", is_half=self.is_half if self.device.type!="privateuseone" else False, device=self.device if self.device.type!="privateuseone"else "cpu"####dml时强制对rmvpe用cpu跑
                #  "rmvpe.pt", is_half=False, device=self.device####dml配置
                # "rmvpe.pt", is_half=False, device="cpu"####锁定cpu配置
                rmvpe_model_path,
                is_half=self.is_half,
                device=self.device,  ####正常逻辑
                use_jit=self.config.use_jit,
            )
            printt("rmvpe model loaded")
            # self.model_rmvpe = RMVPE("aug2_58000_half.pt", is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        feats: torch.Tensor,
        indata: np.ndarray,
        block_frame_16k,
        rate,
        cache_pitch,
        cache_pitchf,
        f0method,
    ) -> np.ndarray:
        feats = feats.view(1, -1)
        if self.config.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        feats = feats.to(self.device)
        t1 = ttime()
        with torch.no_grad():
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.model.extract_features(**inputs)
            feats = (
                self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )
            feats = F.pad(feats, (0, 0, 1, 0))
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                leng_replace_head = int(rate * feats[0].shape[0])
                npy = feats[0][-leng_replace_head:].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                if self.config.is_half:
                    npy = npy.astype("float16")
                feats[0][-leng_replace_head:] = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                    + (1 - self.index_rate) * feats[0][-leng_replace_head:]
                )
            else:
                printt("Index search FAILED or disabled")
        except:
            traceback.print_exc()
            printt("Index search FAILED")
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t3 = ttime()
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(indata, self.f0_up_key, self.n_cpu, f0method)
            start_frame = block_frame_16k // 160
            end_frame = len(cache_pitch) - (pitch.shape[0] - 4) + start_frame
            cache_pitch[:] = np.append(cache_pitch[start_frame:end_frame], pitch[3:-1])
            cache_pitchf[:] = np.append(
                cache_pitchf[start_frame:end_frame], pitchf[3:-1]
            )
            p_len = min(feats.shape[1], 13000, cache_pitch.shape[0])
        else:
            cache_pitch, cache_pitchf = None, None
            p_len = min(feats.shape[1], 13000)
        t4 = ttime()
        feats = feats[:, :p_len, :]
        if self.if_f0 == 1:
            cache_pitch = cache_pitch[:p_len]
            cache_pitchf = cache_pitchf[:p_len]
            cache_pitch = torch.LongTensor(cache_pitch).unsqueeze(0).to(self.device)
            cache_pitchf = torch.FloatTensor(cache_pitchf).unsqueeze(0).to(self.device)
        p_len = torch.LongTensor([p_len]).to(self.device)
        ii = 0  # sid
        sid = torch.LongTensor([ii]).to(self.device)
        with torch.no_grad():
            if self.if_f0 == 1:
                # printt(12222222222,feats.device,p_len.device,cache_pitch.device,cache_pitchf.device,sid.device,rate2)
                infered_audio = self.net_g.infer(
                    feats,
                    p_len,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    torch.FloatTensor([rate]),
                )[0][0, 0].data.float()
            else:
                infered_audio = self.net_g.infer(
                    feats, p_len, sid, torch.FloatTensor([rate])
                )[0][0, 0].data.float()
        t5 = ttime()
        if self.debug:
            printt(
                "Spent time: fea = %.2fs, index = %.2fs, f0 = %.2fs, model = %.2fs",
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio
