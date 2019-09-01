import tensorflow as tf
import sys
import os

from .speaker_score import VoiceScore

vs = VoiceScore(os.path.join(sys.path[0], 'ckpt/Speaker_vox_iter_18000.ckpt'))

score = vs.cal_score(os.path.join(sys.path[0],'tmp/0.wav'), os.path.join(sys.path[0],'tmp/1.wav'))
print(score)