# GRU_SER_VBDRS
AN ENHANCEMENT OF GATED RECURRENT UNIT (GRU) FOR SPEECH EMOTION RECOGNITION IN THE IMPLEMENTATION OF VOICE-BASED DANGER RECOGNITION SYSTEM

Steps for Running:
1. clone repository
2. make sure you are using python 10 not the latest, because tensorflow is not yet compatible with python 12
3. download ffmpeg (for audio file processing): https://github.com/BtbN/FFmpeg-Builds/releases
4. make sure to have the latest visual c++ (for dll errors in tensorflow and keras)
5. create virtual env using python 10
      - python -m venv venv
      - venv\Scripts\activate
      - pip install --upgrade pip
      - pip install setuptools
      - pip install --upgrade pip setuptools
      - python -m ensurepip --upgrade
6. pip install -r requirements.txt or pip install flask pydub numpy tensorflow keras librosa matplotlib
7. run app.py
8. dont commit/push any edits in the app.py, index.html, or requirements.txt if project is in venv.



if venv issues/not showing in terminal:
- Open PowerShell as Administrator:
- Press Windows + X and select Windows PowerShell (Admin) or Terminal (Admin).
- type: Get-ExecutionPolicy
      The result will likely be Restricted.
- type: Set-ExecutionPolicy RemoteSigned
      If prompted, type Y and press Enter to confirm.
- Activate the Virtual Environment: Now try activating the virtual environment again:
      type: venv\Scripts\activate
