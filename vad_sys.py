import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import textgrids

FRAME_DURATION = 30  # 30 msec
OVERLAP_RATE = 0  # frames don't overlap


def readFile(path):
    '''
    Read the file and return the list of SPEECH/NONSPEECH labels for each frame
    '''

    labeled_list = []
    grid = textgrids.TextGrid(path)

    for interval in grid['silences']:
        label = int(interval.text)

        dur = interval.dur
        dur_msec = dur * 1000  # sec -> msec
        num_frames = int(round(dur_msec / 30))  # the audio is divided into 30 msec frames
        print("Duration in miliseconds",dur_msec)
        for i in range(num_frames):
            labeled_list.append(label)

    return labeled_list

def get_audio(name):
    annotation_path = "Data/Annotation/Female/TMIT/"+name+".TextGrid"
    audio_path = "Data/Audio/Female/TMIT/"+name+".wav"
    # read annotaion
    label_list = np.array(readFile(annotation_path))

    # read wav file
    data, sr = librosa.load(audio_path, sr=16000)

    print("sr",sr)

    # define time axis
    Ns = len(data)  # number of sample
    print("Ns",Ns)
    Ts = 1 / sr  # sampling period
    print("Ts",Ts)
    t = np.arange(Ns) * 1000 * Ts  # time axis
    print("t",t)

    shift = 1 - OVERLAP_RATE
    print("shift",shift)
    frame_length = int(np.floor(FRAME_DURATION * sr / 1000)) # frame length in sample
    print("frame_length",frame_length)
    frame_shift = round(frame_length * shift)# frame shift in sample
    print("frame_shift",frame_shift)

    figure = plt.Figure(figsize=(10, 7), dpi=85)
    plt.plot(t, data)

    """    
    for i, frame_labeled in enumerate(label_list):
        idx = i * frame_shift
        if (frame_labeled == 1):
            plt.axvspan(xmin= t[idx], xmax=t[idx + frame_length-1], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='g', label='Speech')"""
    """
    plt.title("Ground truth labels")
    plt.legend(['Signal', 'Speech'])
    plt.show()"""
    return label_list, data, sr


def compute_short_time_energy(audio, frame_size, hop_length):
    """
    Compute short-time energy for the audio signal.

    :param audio: Input audio signal (1D numpy array).
    :param frame_size: Frame size for short-time analysis.
    :param hop_length: Hop length between frames.
    :return: Energy array for each frame.
    """
    # Split the audio into overlapping frames
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_length).T
    print(frames.shape)
    # Calculate short-time energy (sum of squared values per frame)
    energy = np.sum(frames ** 2, axis=1)
    return energy


def energy_thresholding(energy, threshold):
    """
    Apply energy threshold to detect speech activity.

    :param energy: Short-time energy of audio frames.
    :param threshold: Threshold for classifying speech.
    :return: Binary array indicating speech activity (1 for speech, 0 for non-speech).
    """
    return energy > threshold



from scipy.ndimage import median_filter


def smooth_speech_flags(speech_flags, window_size=5):
    """
    Apply smoothing to clean up the detected speech activity flags.

    :param speech_flags: Binary array indicating speech activity.
    :param window_size: Window size for median filtering.
    :return: Smoothed speech activity flags.
    """
    return median_filter(speech_flags, size=window_size)

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_vad(pred_speech_flags, true_speech_flags):
    """
    Evaluate VAD system using precision, recall, and F1-score.

    :param pred_speech_flags: Predicted speech activity (binary array).
    :param true_speech_flags: Ground truth speech activity (binary array).
    :return: Precision, Recall, F1-Score.
    """
    precision = precision_score(true_speech_flags, pred_speech_flags)
    recall = recall_score(true_speech_flags, pred_speech_flags)
    f1 = f1_score(true_speech_flags, pred_speech_flags)

    return precision, recall, f1

precision_all, recall_all, f1_all = [], [], []

data_dir = "Data/Audio/Female/TMIT/"
true_speech_flags, pred_speech_flags = [], []
j = 0
for file_name in os.listdir(data_dir):
    # Check if the file is a .wav file
    if file_name.endswith(".wav"):
        j+=1
        label, audio, sr = get_audio(file_name.replace(".wav", ""))
        # Example of usage
        frame_size = 480  # 30ms frames for 16kHz audio
        hop_length = 480  # 50% overlap
        energy = compute_short_time_energy(audio, frame_size, hop_length)

        """        
        # Plot the energy to visualize it
        plt.figure(figsize=(10, 4))
        plt.plot(energy)
        plt.title("Short-Time Energy")
        plt.show()"""

        # Example usage: set threshold to 1.5 times the mean energy
        threshold = np.mean(energy) * 1.5
        speech_flags = energy_thresholding(energy, threshold)

        """        
        # Plot speech activity
        plt.figure(figsize=(10, 4))
        plt.plot(speech_flags)
        plt.title("Speech Activity Detection (1=Speech, 0=Non-speech)")
        plt.show()"""

        # Example usage
        smoothed_speech_flags = smooth_speech_flags(speech_flags, window_size=7)

        """        
        Plot smoothed speech activity
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed_speech_flags)
        plt.title("Smoothed Speech Activity Detection")
        plt.show()"""

        print("Real label")
        print(label.shape)
        print("predicted smooth Speech flags")
        print((smoothed_speech_flags.astype(int).shape))
        if (label.shape[0] > smoothed_speech_flags.shape[0]):
            smoothed_speech_flags = np.pad(smoothed_speech_flags.astype(int), (label.shape[0]-smoothed_speech_flags.shape[0] , 0), mode='minimum')
        if (label.shape[0] < smoothed_speech_flags.shape[0]):
            label = np.pad(label,(smoothed_speech_flags.shape[0] - label.shape[0], 0), mode='minimum')
        print((smoothed_speech_flags.astype(int).shape))
        #true_speech_flags.append(label)
        #pred_speech_flags.append(smoothed_speech_flags.astype(int))

        # Example usage (assume you have true_speech_flags)
        # true_speech_flags = [...]  # Ground truth labels
        precision, recall, f1 = evaluate_vad(label, smoothed_speech_flags)

        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f1)



# Print evaluation metrics
print(f"Precision: {sum(precision_all)/j:.2f}, Recall: {sum(recall_all)/j:.2f}, F1-Score: {sum(f1_all)/j:.2f}")
