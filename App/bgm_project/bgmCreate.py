from transformers import pipeline
import scipy

#設定　音楽を生成するために変数を設定
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small",model_kwargs={"attn_implementation": "eager"})
#　音楽を生成させる命令 txtの部分を変更することによって、音楽を生成する命令を変えれる
music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})
# 音楽を保持する
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
