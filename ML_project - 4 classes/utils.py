import imports

def create_fft(fn):
    if not imports.os.path.exists(fn):
        return 
    sample_rate, X = imports.scipy.io.wavfile.read(fn)
    if len(X.shape) > 1:
        X = X.mean(axis=1)
    fft_features = abs(imports.scipy.fft.fft(X)[:20]) 
    return fft_features


def calculeFeautures(audio_path):
    y, sr = imports.librosa.load(audio_path, mono=True, duration=30) 
    chroma_stft = imports.librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = imports.librosa.feature.rms(y=y)
    spec_cent = imports.librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = imports.librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = imports.librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = imports.librosa.feature.zero_crossing_rate(y)
    mfcc = imports.librosa.feature.mfcc(y=y, sr=sr)[:10]
    D = imports.np.abs(imports.librosa.stft(y))**2  # Spectrogramme de puissance
    log_spectrum = imports.np.log(imports.np.maximum(D, 1e-10))  # Éviter les log(0)
    rceps = imports.np.real(imports.np.fft.ifft(log_spectrum, axis=0))  # Transformée de Fourier inverse
    rceps_mean = imports.np.mean(rceps[:10], axis=1)
    fft_features = create_fft(audio_path)
    features= [float(imports.np.mean(chroma_stft)),float(imports.np.mean(rmse)),float(imports.np.mean(spec_cent)),float(imports.np.mean(spec_bw)),float(imports.np.mean(rolloff)),float(imports.np.mean(zcr))]
    features.extend([float(imports.np.mean(mfcc_coef)) for mfcc_coef in mfcc])

    # Ajouter les moyennes des coefficients cepstraux
    features.extend([float(value) for value in rceps_mean])

    # Ajouter les caractéristiques FFT
    if fft_features is not None:
        features.extend([float(value) for value in fft_features])
    return features

def sanitize_filename(filename):
    return imports.re.sub(r'[\\/*?:"<>|]', "_", filename)


def download_and_extract_audio(youtube_url, output_path="audios/"):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": f"{output_path}%(title)s.%(ext)s",
        "ffmpeg_location": "ffmpeg-2024-12-11-git-a518b5540d-full_build/ffmpeg-2024-12-11-git-a518b5540d-full_build/bin",
        "postprocessor_args": [
            "-ss", "00:01:30", 
            "-t", "00:01:00"  
        ],
    }

    with imports.yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    
    mp3_filename = sanitize_filename(info['title']) + ".mp3"
    wav_filename = sanitize_filename(info['title']) + ".wav"
    mp3_path = f"{output_path}{mp3_filename}"
    wav_path = f"{output_path}{wav_filename}"
    
    #mp3_path = f"{output_path}{info['title']}.mp3"
    audio = imports.AudioSegment.from_mp3(mp3_path)  # Charger le MP3
    #wav_path = f"{output_path}{info['title']}.wav"  # Définir le chemin pour le fichier WAV

    audio = imports.normalize(audio)
    audio.export(wav_path, format="wav")
    return wav_path


def search_youtube(genre, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    api_key = "AIzaSyA_l0nnMWshlevTsrO0j9GjDLuFu_3lrF8"  # Remplacez par votre clé API YouTube
    params = {
        "part": "snippet",
        "q": genre + " music",  # Requête basée sur le genre
        "type": "video",
        "maxResults": max_results,
        "key": api_key
    }
    
    response = imports.requests.get(search_url, params=params)
    results = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get("items", []):
            video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            results.append(video_url)
    else:
        print(f"Erreur lors de l'appel à l'API YouTube : {response.status_code}")
    
    return results
    



    