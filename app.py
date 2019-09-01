import os
import time
from flask import Flask, request, flash, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap

from speaker_score.speaker_score import VoiceScore


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'wav'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
bootstrap = Bootstrap(app)
vs = VoiceScore(os.path.join(os.getcwd(), 'speaker_score/ckpt/Speaker_vox_iter_18000.ckpt'))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('hello.html')


@app.route('/voice_check', methods=['GET', 'POST'])
def voice_check():
    form = VoiceInputForm()
    if form.validate_on_submit():
        voc1 = request.files[form.voice1.name]
        voc2 = request.files[form.voice2.name]
        secure_voc1_name = secure_save_file(voc1)
        secure_voc2_name = secure_save_file(voc2)
        if not secure_voc1_name or not secure_voc2_name:
            return render_template('index.html', form=form)
        voc1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_voc1_name)
        voc2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_voc2_name)
        score = vs.cal_score(voc1_path, voc2_path)
        return render_template('result.html',
                               voc1_name=secure_voc1_name,
                               voc2_name=secure_voc2_name,
                               score=score)
    return render_template('index.html', form=form)


class VoiceInputForm(FlaskForm):
    voice1 = FileField(u'输入一个 wav 声音文件', validators=[
        DataRequired(u'请输入一个有效的声音文件')
    ])
    voice2 = FileField(u'输入另一个 wav 声音文件', validators=[
        DataRequired(u'请输入一个有效的声音文件')
    ])
    submit = SubmitField(u'提交')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_save_file(file):
    if file and allowed_file(file.filename):
        cur_time = '-'.join(time.ctime().split(' ')).replace(':', '-')
        filename = cur_time + '_' + secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename
    else:
        flash('文件不符合要求')
        return None


if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
