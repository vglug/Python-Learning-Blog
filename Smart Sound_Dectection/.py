import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile
import warnings
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# AUDIO GENERATOR - Creates test audio files
# ============================================================================

class TestAudioGenerator:
    """Generate synthetic test audio with claps and whistles"""
    
    @staticmethod
    def generate_clap(sr=22050, duration=0.1):
        """Generate synthetic clap sound"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Clap: short burst of noise with percussive envelope
        noise = np.random.normal(0, 0.3, len(t))
        
        # Percussive envelope (quick attack, fast decay)
        envelope = np.exp(-35 * t)
        
        clap = noise * envelope
        return clap
    
    @staticmethod
    def generate_whistle(sr=22050, duration=0.5, frequency=1000):
        """Generate synthetic whistle sound"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Whistle: sine wave with amplitude envelope
        whistle = 0.2 * np.sin(2 * np.pi * frequency * t)
        
        # Smooth envelope (fade in, fade out)
        envelope = np.sin(np.pi * t / duration) ** 2
        
        whistle = whistle * envelope
        return whistle
    
    @staticmethod
    def generate_background_noise(sr=22050, duration=3, snr_db=15):
        """Generate background noise"""
        noise = np.random.normal(0, 0.05, int(sr * duration))
        return noise
    
    @staticmethod
    def create_test_audio(output_file="test_audio.wav", sr=22050):
        """Create complete test audio file with claps and whistles"""
        print(f"\n🎵 Generating test audio file: {output_file}")
        
        duration = 5  # 5 seconds total
        samples = int(sr * duration)
        audio = np.zeros(samples)
        
        # Background noise
        noise = TestAudioGenerator.generate_background_noise(sr, duration, snr_db=20)
        audio += noise
        
        # Add claps at different times
        clap_times = [0.5, 1.2, 2.0, 3.5, 4.2]
        for clap_time in clap_times:
            clap_start = int(clap_time * sr)
            clap = TestAudioGenerator.generate_clap(sr)
            clap_end = clap_start + len(clap)
            audio[clap_start:clap_end] += clap
        
        # Add whistles at different times
        whistle_times = [(0.8, 800), (1.5, 1200), (2.5, 1000), (3.2, 900)]
        for whistle_time, freq in whistle_times:
            whistle_start = int(whistle_time * sr)
            whistle = TestAudioGenerator.generate_whistle(sr, duration=0.5, frequency=freq)
            whistle_end = whistle_start + len(whistle)
            audio[whistle_start:whistle_end] += whistle
        
        # Normalize and prevent clipping
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
        
        # Save as WAV file
        audio_int16 = np.int16(audio * 32767)
        wavfile.write(output_file, sr, audio_int16)
        
        print(f"✅ Test audio generated: {output_file}")
        return output_file

# ============================================================================
# ADVANCED SOUND EVENT DETECTOR
# ============================================================================

class AdvancedSoundEventDetector:
    """Ultra-Advanced Sound Event Detection System"""
    
    def __init__(self, sr=22050, frame_length=2048, hop_length=512):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.clap_threshold = 0.55
        self.whistle_threshold = 0.50
        self.detection_history = []
    
    def load_audio(self, file_path):
        """Load audio file"""
        try:
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")
                return None, None
            
            y, sr = librosa.load(file_path, sr=self.sr)
            print(f"✅ Audio loaded: {len(y)/sr:.2f}s @ {sr}Hz")
            return y, sr
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None, None
    
    def extract_features(self, y):
        """Extract comprehensive audio features"""
        print("\n📊 Extracting audio features...")
        
        # Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.frame_length,
                                          hop_length=self.hop_length, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length,
                                                 hop_length=self.hop_length)
        
        # Spectral Features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        
        # Onset Detection
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length,
                                  hop_length=self.hop_length)
        
        # Harmonic-Percussive Source Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_strength = librosa.feature.rms(y=y_harmonic, frame_length=self.frame_length,
                                               hop_length=self.hop_length)
        percussive_strength = librosa.feature.rms(y=y_percussive, frame_length=self.frame_length,
                                                 hop_length=self.hop_length)
        
        # Spectral Flux
        spec_flux = np.sqrt(np.sum(np.diff(S_db, axis=1)**2, axis=0))
        
        # Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_energy = np.sqrt(np.sum(chroma**2, axis=0))
        
        # Spectral Variation
        spectral_var = np.std(S_db, axis=0)
        
        print("✅ Features extracted")
        
        return {
            'melspec': S_db,
            'mfcc': mfcc,
            'zcr': zcr,
            'spec_centroid': spec_centroid,
            'spec_rolloff': spec_rolloff,
            'spec_bandwidth': spec_bandwidth,
            'onset_env': onset_env,
            'rms': rms,
            'harmonic_strength': harmonic_strength,
            'percussive_strength': percussive_strength,
            'spectral_flux': spec_flux,
            'spec_contrast': spec_contrast,
            'chroma_energy': chroma_energy,
            'spectral_var': spectral_var,
            'y_harmonic': y_harmonic,
            'y_percussive': y_percussive
        }
    
    def detect_clap(self, y, features):
        """Detect clap sounds"""
        print("\n🔍 Detecting claps...")
        
        zcr = features['zcr'][0]
        onset_env = features['onset_env']
        percussive = features['percussive_strength'][0]
        
        # Normalize features
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)
        percussive_norm = (percussive - np.mean(percussive)) / (np.std(percussive) + 1e-8)
        
        # Find peaks in onset
        peaks, properties = signal.find_peaks(onset_env, 
                                             height=np.max(onset_env)*0.25,
                                             distance=int(self.sr/self.hop_length*0.1))
        
        clap_candidates = []
        
        for peak in peaks:
            # Clap scoring
            clap_score = (
                0.4 * (1 if zcr_norm[peak] > 0.4 else 0) +
                0.4 * (1 if percussive_norm[peak] > 0.2 else 0) +
                0.2 * (properties['peak_heights'][list(peaks).index(peak)] / np.max(onset_env))
            )
            
            if clap_score > self.clap_threshold:
                clap_candidates.append((peak, clap_score))
        
        clap_frames = self._filter_close_detections(clap_candidates, 
                                                    min_distance=int(0.3*self.sr/self.hop_length))
        
        print(f"✅ Found {len(clap_frames)} clap(s)")
        return clap_frames
    
    def detect_whistle(self, y, features):
        """Detect whistle sounds"""
        print("\n🔍 Detecting whistles...")
        
        zcr = features['zcr'][0]
        mfcc = features['mfcc']
        harmonic = features['harmonic_strength'][0]
        chroma_energy = features['chroma_energy']
        
        # Normalize
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)
        harmonic_norm = (harmonic - np.mean(harmonic)) / (np.std(harmonic) + 1e-8)
        
        # MFCC stability
        mfcc_var = np.var(mfcc, axis=0)
        mfcc_var_norm = (mfcc_var - np.mean(mfcc_var)) / (np.std(mfcc_var) + 1e-8)
        
        whistle_candidates = []
        
        for i in range(len(zcr)):
            # Whistle scoring
            whistle_score = (
                0.3 * (1 if zcr_norm[i] < -0.3 else 0) +
                0.3 * (1 if harmonic_norm[i] > 0.2 else 0) +
                0.25 * (1 if mfcc_var_norm[i] < 0.2 else 0) +
                0.15 * (1 if chroma_energy[i] > np.mean(chroma_energy) else 0)
            )
            
            if whistle_score > self.whistle_threshold:
                whistle_candidates.append((i, whistle_score))
        
        whistle_frames = self._cluster_frames(whistle_candidates, 
                                             min_duration=int(0.1*self.sr/self.hop_length))
        
        print(f"✅ Found {len(whistle_frames)} whistle(s)")
        return whistle_frames
    
    def _filter_close_detections(self, candidates, min_distance):
        """Filter duplicate detections"""
        if not candidates:
            return []
        
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        filtered = []
        
        for frame, score in candidates:
            if not any(abs(frame - f[0]) < min_distance for f in filtered):
                filtered.append((frame, score))
        
        return sorted(filtered, key=lambda x: x[0])
    
    def _cluster_frames(self, candidates, min_duration):
        """Cluster consecutive frames"""
        if not candidates:
            return []
        
        candidates = sorted(candidates, key=lambda x: x[0])
        clusters = []
        current_cluster = [candidates[0]]
        
        for i in range(1, len(candidates)):
            if candidates[i][0] - current_cluster[-1][0] < min_duration:
                current_cluster.append(candidates[i])
            else:
                best_frame = max(current_cluster, key=lambda x: x[1])
                clusters.append(best_frame)
                current_cluster = [candidates[i]]
        
        if current_cluster:
            best_frame = max(current_cluster, key=lambda x: x[1])
            clusters.append(best_frame)
        
        return sorted(clusters, key=lambda x: x[0])
    
    def detect(self, file_path, plot=True, save_results=False):
        """Main detection pipeline"""
        print("\n" + "="*70)
        print("🎵 ADVANCED SOUND EVENT DETECTION SYSTEM")
        print("="*70)
        
        y, sr = self.load_audio(file_path)
        if y is None:
            return None
        
        features = self.extract_features(y)
        
        clap_frames = self.detect_clap(y, features)
        whistle_frames = self.detect_whistle(y, features)
        
        clap_times = librosa.frames_to_time(np.array([f[0] for f in clap_frames]),
                                           sr=self.sr, hop_length=self.hop_length)
        whistle_times = librosa.frames_to_time(np.array([f[0] for f in whistle_frames]),
                                              sr=self.sr, hop_length=self.hop_length)
        
        results = {
            'clap': clap_frames,
            'whistle': whistle_frames,
            'clap_times': clap_times,
            'whistle_times': whistle_times,
            'features': features,
            'audio': y,
            'file': file_path,
            'timestamp': datetime.now().isoformat()
        }
        
        self.detection_history.append(results)
        
        if plot:
            self.plot_results(y, features, clap_frames, whistle_frames)
        
        if save_results:
            self.save_results(results)
        
        self.print_summary(results)
        
        return results
    
    def plot_results(self, y, features, clap_frames, whistle_frames):
        """Visualize detection results"""
        print("\n📈 Generating visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
        
        frames_axis = np.arange(len(features['zcr'][0]))
        time_axis = librosa.frames_to_time(frames_axis, sr=self.sr, hop_length=self.hop_length)
        
        # 1. Waveform
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(np.arange(len(y))/self.sr, y, alpha=0.7, linewidth=0.8)
        ax1.set_title('Audio Waveform', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mel Spectrogram
        ax2 = fig.add_subplot(gs[1, :])
        img = librosa.display.specshow(features['melspec'], sr=self.sr, 
                                       hop_length=self.hop_length, x_axis='time',
                                       y_axis='mel', ax=ax2)
        ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
        plt.colorbar(img, ax=ax2, label='dB')
        
        # 3. ZCR
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(time_axis, features['zcr'][0], label='ZCR', alpha=0.7)
        if len(clap_frames) > 0:
            clap_times = librosa.frames_to_time(np.array([f[0] for f in clap_frames]),
                                               sr=self.sr, hop_length=self.hop_length)
            ax3.scatter(clap_times, features['zcr'][0][[f[0] for f in clap_frames]], 
                       color='red', s=100, marker='o', label='Clap', zorder=5)
        ax3.set_title('Zero Crossing Rate', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Spectral Features
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(time_axis, features['spec_centroid'][0], label='Centroid', alpha=0.7)
        ax4.plot(time_axis, features['spec_rolloff'][0], label='Rolloff', alpha=0.7)
        ax4.set_title('Spectral Features', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Harmonic vs Percussive
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(time_axis, features['harmonic_strength'][0], label='Harmonic', alpha=0.7)
        ax5.plot(time_axis, features['percussive_strength'][0], label='Percussive', alpha=0.7)
        if len(whistle_frames) > 0:
            whistle_times = librosa.frames_to_time(np.array([f[0] for f in whistle_frames]),
                                                  sr=self.sr, hop_length=self.hop_length)
            ax5.scatter(whistle_times, features['harmonic_strength'][0][[f[0] for f in whistle_frames]], 
                       color='green', s=100, marker='^', label='Whistle', zorder=5)
        ax5.set_title('Harmonic vs Percussive', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Onset Strength
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.plot(time_axis, features['onset_env'], label='Onset Envelope', alpha=0.7, color='orange')
        ax6.set_title('Onset Strength', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. MFCC
        ax7 = fig.add_subplot(gs[4, :])
        img_mfcc = librosa.display.specshow(features['mfcc'], sr=self.sr,
                                           hop_length=self.hop_length, x_axis='time', ax=ax7)
        ax7.set_title('MFCC Features', fontsize=11, fontweight='bold')
        plt.colorbar(img_mfcc, ax=ax7)
        
        plt.suptitle('Advanced Sound Event Detection Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        print("✅ Visualization complete")
    
    def print_summary(self, results):
        """Print detection summary"""
        print("\n" + "="*70)
        print("📋 DETECTION RESULTS")
        print("="*70)
        
        print(f"\n🔊 CLAPS DETECTED: {len(results['clap'])}")
        if len(results['clap']) > 0:
            for i, (clap_data, time) in enumerate(zip(results['clap'], 
                                                       results['clap_times']), 1):
                confidence = clap_data[1] * 100 if isinstance(clap_data, tuple) else 80
                print(f"   Clap {i}: {time:.3f}s | Confidence: {confidence:.1f}%")
        
        print(f"\n🎵 WHISTLES DETECTED: {len(results['whistle'])}")
        if len(results['whistle']) > 0:
            for i, (whistle_data, time) in enumerate(zip(results['whistle'], 
                                                         results['whistle_times']), 1):
                confidence = whistle_data[1] * 100 if isinstance(whistle_data, tuple) else 80
                print(f"   Whistle {i}: {time:.3f}s | Confidence: {confidence:.1f}%")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Events: {len(results['clap']) + len(results['whistle'])}")
        print(f"   Duration: {len(results['audio'])/self.sr:.2f}s")
        print("="*70 + "\n")
    
    def save_results(self, results, output_file="detection_results.json"):
        """Save results to JSON"""
        save_data = {
            'file': results['file'],
            'timestamp': results['timestamp'],
            'duration': len(results['audio']) / self.sr,
            'claps': [{'time': float(t), 'confidence': float(results['clap'][i][1]*100)} 
                     for i, t in enumerate(results['clap_times'])],
            'whistles': [{'time': float(t), 'confidence': float(results['whistle'][i][1]*100)} 
                        for i, t in enumerate(results['whistle_times'])]
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"✅ Results saved to {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Generate test audio
    print("\n" + "🎯 STEP 1: Generate Test Audio")
    audio_generator = TestAudioGenerator()
    test_file = audio_generator.create_test_audio("test_audio.wav")
    
    # Step 2: Initialize detector
    print("\n" + "🎯 STEP 2: Initialize Detector")
    detector = AdvancedSoundEventDetector(sr=22050)
    
    # Step 3: Run detection
    print("\n" + "🎯 STEP 3: Run Detection")
    results = detector.detect(test_file, plot=True, save_results=True)
    
    print("\n✨ DETECTION COMPLETE!")
    print("📁 Generated files:")
    print("   - test_audio.wav (audio file)")
    print("   - detection_results.json (results)")
